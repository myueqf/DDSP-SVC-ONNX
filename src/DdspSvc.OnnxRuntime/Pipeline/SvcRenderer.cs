using DdspSvc.OnnxRuntime.Audio;
using DdspSvc.OnnxRuntime.Models;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class SvcRenderer {
    private readonly SvcInferencePipeline pipeline;

    public SvcRenderer(SvcInferencePipeline pipeline) {
        this.pipeline = pipeline;
    }

    public SvcInferenceResult Render(SvcInferenceRequest request) {
        var conditioning = PrepareConditioning(request);
        return Render(conditioning);
    }

    public void Validate(SvcInferenceRequest request) {
        Analyze(request);
    }

    public SvcRenderInfo Analyze(SvcInferenceRequest request) {
        ArgumentNullException.ThrowIfNull(request);
        if (request.SampleRate <= 0) {
            throw new ArgumentOutOfRangeException(nameof(request), "SampleRate must be positive.");
        }

        ValidateRuntimeCapabilities(request);

        var reflowSteps = request.ReflowSteps ?? pipeline.Options.ReflowSteps;
        if (reflowSteps <= 0) {
            throw new ArgumentOutOfRangeException(nameof(request), "ReflowSteps must be positive.");
        }

        var targetSampleCount = GetResampledSampleCount(
            request.Audio.Length,
            request.SampleRate,
            pipeline.VocoderConfig.SampleRate);
        var totalFrames = GetFrameCount(targetSampleCount, pipeline.Options.HopSize);
        var segments = request.Audio.Length == 0
            ? 0
            : BuildSegments(
                request.Audio,
                request.SampleRate,
                pipeline.VocoderConfig.SampleRate,
                totalFrames).Count;

        return new SvcRenderInfo(
            InputSampleRate: request.SampleRate,
            OutputSampleRate: pipeline.VocoderConfig.SampleRate,
            InputSamples: request.Audio.Length,
            OutputSamples: targetSampleCount,
            TotalFrames: totalFrames,
            SegmentCount: segments,
            ReflowSteps: reflowSteps,
            KeyShiftSemitones: request.KeyShiftSemitones,
            SilenceThresholdDb: request.SilenceThresholdDb ?? pipeline.Options.SilenceThresholdDb,
            UsedSpeakerMix: request.SpeakerMix is { Length: > 0 });
    }

    public SvcPreparedConditioning PrepareConditioning(SvcInferenceRequest request) {
        var analysis = Analyze(request);
        var speakerMix = NormalizeSpeakerMix(request.SpeakerMix, pipeline.Options.SpeakerCount);
        if (request.Audio.Length == 0) {
            return new SvcPreparedConditioning {
                InputSampleRate = request.SampleRate,
                OutputSampleRate = pipeline.VocoderConfig.SampleRate,
                HopSize = pipeline.Options.HopSize,
                Seed = request.Seed,
                ReflowSteps = analysis.ReflowSteps,
                KeyShiftSemitones = request.KeyShiftSemitones,
                SilenceThresholdDb = analysis.SilenceThresholdDb,
                TotalFrames = 0,
                VolumeMask = [],
                SpeakerMix = speakerMix,
                Segments = [],
                RenderInfo = analysis with { OutputSamples = 0, SegmentCount = 0 },
            };
        }

        var pitch = pipeline.ExtractPitch(request.Audio, request.SampleRate);
        var volume = pipeline.ExtractVolume(request.Audio, request.SampleRate);
        var totalFrames = Math.Min(pitch.F0Hz.Length, volume.Length);
        var targetSampleCount = Math.Min(
            GetResampledSampleCount(request.Audio.Length, request.SampleRate, pipeline.VocoderConfig.SampleRate),
            Math.Max(0, totalFrames * pipeline.Options.HopSize));
        var conditionedPitch = PitchConditioning.InterpolateUnvoiced(pitch.F0Hz.Take(totalFrames).ToArray(), pipeline.Options.F0MinHz);
        if (request.KeyShiftSemitones != 0) {
            var factor = MathF.Pow(2f, request.KeyShiftSemitones / 12f);
            for (var i = 0; i < conditionedPitch.Length; i++) {
                conditionedPitch[i] *= factor;
            }
        }
        volume = volume.Take(totalFrames).ToArray();
        var mask = VolumeMask.BuildBinary(volume, pipeline.Options.HopSize, targetSampleCount, analysis.SilenceThresholdDb);
        var baseSegments = BuildSegments(
            request.Audio,
            request.SampleRate,
            pipeline.VocoderConfig.SampleRate,
            totalFrames);
        if (baseSegments.Count == 0) {
            baseSegments.Add(new Segment(0, totalFrames, 0, request.Audio.Length));
        }

        var segments = new List<SvcSegmentConditioning>(baseSegments.Count);
        foreach (var segment in baseSegments) {
            var inputStartSample = Math.Clamp(segment.InputStartSample, 0, request.Audio.Length);
            var inputEndSample = Math.Clamp(segment.InputEndSample, inputStartSample, request.Audio.Length);
            var segmentAudio = request.Audio[inputStartSample..inputEndSample];
            var units = pipeline.ExtractUnits(segmentAudio, request.SampleRate);
            var availableFrames = new[] { units.Frames, segment.EndFrame - segment.StartFrame }.Min();
            if (availableFrames <= 0 || segment.StartFrame + availableFrames > totalFrames) {
                continue;
            }

            segments.Add(new SvcSegmentConditioning {
                StartFrame = segment.StartFrame,
                EndFrame = segment.StartFrame + availableFrames,
                StartSample = segment.StartFrame * pipeline.Options.HopSize,
                EndSample = Math.Min(targetSampleCount, (segment.StartFrame + availableFrames) * pipeline.Options.HopSize),
                Units = new ContentUnitsResult {
                    Frames = availableFrames,
                    Channels = units.Channels,
                    Units = units.Units.Take(availableFrames * units.Channels).ToArray(),
                },
                F0Hz = conditionedPitch.Skip(segment.StartFrame).Take(availableFrames).ToArray(),
                Volume = volume.Skip(segment.StartFrame).Take(availableFrames).ToArray(),
            });
        }

        return new SvcPreparedConditioning {
            InputSampleRate = request.SampleRate,
            OutputSampleRate = pipeline.VocoderConfig.SampleRate,
            HopSize = pipeline.Options.HopSize,
            Seed = request.Seed,
            ReflowSteps = analysis.ReflowSteps,
            KeyShiftSemitones = request.KeyShiftSemitones,
            SilenceThresholdDb = analysis.SilenceThresholdDb,
            TotalFrames = totalFrames,
            VolumeMask = mask,
            SpeakerMix = speakerMix,
            Segments = segments,
            RenderInfo = analysis with {
                TotalFrames = totalFrames,
                SegmentCount = segments.Count,
                OutputSamples = targetSampleCount,
            },
        };
    }

    public SvcInferenceResult Render(SvcPreparedConditioning conditioning) {
        ArgumentNullException.ThrowIfNull(conditioning);
        if (conditioning.Segments.Count == 0) {
            return new SvcInferenceResult {
                Audio = [],
                SampleRate = pipeline.VocoderConfig.SampleRate,
                RenderInfo = conditioning.RenderInfo with { OutputSamples = 0, SegmentCount = 0 },
            };
        }

        var result = Array.Empty<float>();
        var currentLength = 0;
        foreach (var segment in conditioning.Segments) {
            var ddsp = pipeline.EncodeDdsp(
                segment.Units,
                segment.F0Hz,
                segment.Volume,
                conditioning.Seed,
                conditioning.SpeakerMix);

            var inputPitch = segment.F0Hz.Take(ddsp.Frames).ToArray();
            var melNormalized = pipeline.SampleReflow(ddsp, conditioning.ReflowSteps, conditioning.Seed);
            var melFrameMajor = MelProjection.FlattenToFrameMajor(melNormalized, ddsp.Frames, ddsp.MelBins);
            var melForVocoder = MelProjection.Denormalize(melFrameMajor, pipeline.VocoderConfig.MelBase);
            var vocoder = pipeline.VocoderInfer(melForVocoder, ddsp.Frames, ddsp.MelBins, inputPitch);

            ApplyMaskSegment(vocoder.Audio, conditioning.VolumeMask, segment.StartSample);
            result = Stitch(result, vocoder.Audio, segment.StartSample, ref currentLength);
        }

        return new SvcInferenceResult {
            Audio = result,
            SampleRate = pipeline.VocoderConfig.SampleRate,
            RenderInfo = conditioning.RenderInfo with {
                OutputSamples = result.Length,
                SegmentCount = conditioning.Segments.Count,
            },
        };
    }

    private List<Segment> BuildSegments(
        float[] audio,
        int inputSampleRate,
        int targetSampleRate,
        int totalFrames) {
        var inputHopSize = pipeline.Options.HopSize * inputSampleRate / (double)targetSampleRate;
        var slicer = new AudioSlicer(inputSampleRate);
        return slicer.Slice(audio)
            .Where(slice => !slice.IsSilence && slice.EndSample > slice.StartSample)
            .Select(slice => {
                var startFrame = Math.Min(totalFrames, (int)(slice.StartSample / inputHopSize));
                var endFrame = Math.Min(totalFrames, (int)(slice.EndSample / inputHopSize));
                var inputStartSample = Math.Clamp((int)(startFrame * inputHopSize), 0, audio.Length);
                var inputEndSample = Math.Clamp((int)(endFrame * inputHopSize), inputStartSample, audio.Length);
                return new Segment(startFrame, endFrame, inputStartSample, inputEndSample);
            })
            .Where(segment =>
                segment.EndFrame > segment.StartFrame &&
                segment.InputEndSample > segment.InputStartSample)
            .ToList();
    }

    private static int GetFrameCount(int sampleCount, int hopSize) {
        if (sampleCount <= 0) {
            return 0;
        }
        return sampleCount / hopSize + 1;
    }

    private static int GetResampledSampleCount(int sourceSampleCount, int sourceSampleRate, int targetSampleRate) {
        if (sourceSampleCount <= 0) {
            return 0;
        }
        return Math.Max(1, (int)Math.Round(sourceSampleCount * (double)targetSampleRate / sourceSampleRate));
    }

    private void ValidateRuntimeCapabilities(SvcInferenceRequest request) {
        if (!pipeline.Capabilities.SupportedReflowMethods.Contains(pipeline.Options.ReflowMethod, StringComparer.OrdinalIgnoreCase)) {
            throw new UnsupportedSvcFeatureException(
                "reflow.method",
                $"Reflow method '{pipeline.Options.ReflowMethod}' is not supported yet.");
        }
        if (Math.Abs(pipeline.Options.ReflowTStart) > 1e-6f && !pipeline.Capabilities.SupportsNonZeroTStart) {
            throw new UnsupportedSvcFeatureException("reflow.t_start", "Non-zero t_start is not supported yet.");
        }
        if (request.SpeakerMix is { Length: > 0 }) {
            if (!pipeline.Capabilities.SupportsSpeakerMix) {
                throw new UnsupportedSvcFeatureException("speaker_mix", "Speaker mixing requires a multi-speaker ONNX export.");
            }
            if (request.SpeakerMix.Length != pipeline.Capabilities.SpeakerCount) {
                throw new ArgumentException(
                    $"Speaker mix length must equal model speaker count ({pipeline.Capabilities.SpeakerCount}).",
                    nameof(request));
            }
        }
        if (request.FormantShiftSemitones != 0 && !pipeline.Capabilities.SupportsFormantShift) {
            throw new UnsupportedSvcFeatureException(
                "formant_shift_key",
                "The current ONNX export does not expose aug_shift, so formant shifting is not supported yet.");
        }
        if (request.VocalRegisterShiftSemitones != 0 && !pipeline.Capabilities.SupportsVocalRegisterShift) {
            throw new UnsupportedSvcFeatureException(
                "vocal_register_shift_key",
                "The current ONNX export does not expose aug_shift, so vocal register shifting is not supported yet.");
        }
    }

    private static void ApplyMaskSegment(float[] audio, float[] mask, int startSample) {
        var length = Math.Min(audio.Length, Math.Max(0, mask.Length - startSample));
        for (var i = 0; i < length; i++) {
            audio[i] *= mask[startSample + i];
        }
    }

    private static float[] Stitch(float[] existing, float[] segmentAudio, int startSample, ref int currentLength) {
        var silentLength = startSample - currentLength;
        if (silentLength >= 0) {
            var result = new float[currentLength + silentLength + segmentAudio.Length];
            Array.Copy(existing, result, existing.Length);
            Array.Copy(segmentAudio, 0, result, currentLength + silentLength, segmentAudio.Length);
            currentLength += silentLength + segmentAudio.Length;
            return result;
        }

        return CrossFade(existing, segmentAudio, currentLength + silentLength, ref currentLength);
    }

    private static float[] CrossFade(float[] a, float[] b, int idx, ref int currentLength) {
        var result = new float[idx + b.Length];
        var fadeLength = a.Length - idx;
        Array.Copy(a, result, idx);
        for (var i = 0; i < fadeLength; i++) {
            var k = fadeLength <= 1 ? 1f : i / (float)(fadeLength - 1);
            result[idx + i] = (1f - k) * a[idx + i] + k * b[i];
        }
        Array.Copy(b, fadeLength, result, a.Length, b.Length - fadeLength);
        currentLength = result.Length;
        return result;
    }

    private static float[]? NormalizeSpeakerMix(float[]? speakerMix, int speakerCount) {
        if (speakerCount <= 1) {
            return null;
        }
        if (speakerMix is null || speakerMix.Length == 0) {
            var defaultMix = new float[speakerCount];
            defaultMix[0] = 1f;
            return defaultMix;
        }

        if (speakerMix.Any(value => value < 0f || float.IsNaN(value) || float.IsInfinity(value))) {
            throw new ArgumentException("Speaker mix values must be finite and non-negative.", nameof(speakerMix));
        }
        var sum = speakerMix.Sum();
        if (sum <= 0f) {
            throw new ArgumentException("Speaker mix must contain at least one positive weight.", nameof(speakerMix));
        }
        return speakerMix.Select(value => value / sum).ToArray();
    }

    private readonly record struct Segment(int StartFrame, int EndFrame, int InputStartSample, int InputEndSample);
}

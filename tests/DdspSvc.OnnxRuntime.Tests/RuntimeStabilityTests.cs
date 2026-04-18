using DdspSvc.OnnxRuntime.Audio;
using DdspSvc.OnnxRuntime.Hosting;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Pipeline;

namespace DdspSvc.OnnxRuntime.Tests;

public class RuntimeStabilityTests {
    [Fact]
    public void VolumeExtractor_ReturnsExpectedFrameCount() {
        var samples = Enumerable.Range(0, 4096).Select(i => (float)Math.Sin(i * 0.1)).ToArray();
        var extractor = new VolumeExtractor(hopSize: 512, winSize: 2048);

        var volume = extractor.Extract(samples);

        Assert.Equal(samples.Length / 512 + 1, volume.Length);
        Assert.All(volume, value => Assert.True(value >= 0));
    }

    [Fact]
    public void WaveFile_StreamReaderAndWriter_RoundTripSamples() {
        var root = CreateTempDirectory();
        try {
            var path = Path.Combine(root, "roundtrip.wav");
            var first = Enumerable.Range(0, 1000).Select(i => MathF.Sin(i * 0.01f) * 0.25f).ToArray();
            var second = Enumerable.Range(0, 750).Select(i => MathF.Cos(i * 0.02f) * 0.15f).ToArray();

            using (var writer = WaveFile.OpenWriter(path, 44100)) {
                writer.WriteMonoSamples(first);
                writer.WriteMonoSamples(second);
            }

            using var reader = WaveFile.OpenReader(path);
            var a = reader.ReadMonoSamples(600);
            var b = reader.ReadMonoSamples(2000);

            Assert.Equal(44100, reader.SampleRate);
            Assert.Equal(1, reader.Channels);
            Assert.Equal(first.Length + second.Length, a.Length + b.Length);
            Assert.Equal(first.Concat(second).ToArray(), a.Concat(b).ToArray(), new FloatArrayComparer(2e-4f));
            Assert.Empty(reader.ReadMonoSamples(1));
        } finally {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void PitchConditioning_InterpolatesAndClampsUnvoicedFrames() {
        var f0 = new float[] { 0f, 220f, 0f, 0f, 440f, 0f };

        var conditioned = PitchConditioning.InterpolateUnvoiced(f0, minF0Hz: 65f);

        Assert.Equal(new float[] { 220f, 220f, 293.33334f, 366.66666f, 440f, 440f }, conditioned, new FloatArrayComparer(1e-3f));
    }

    [Fact]
    public void VolumeMaskBuildBinary_UpsamplesFrameMaskToAudioLength() {
        var mask = VolumeMask.BuildBinary(new[] { 0f, 0.02f, 0.2f }, hopSize: 4, targetLength: 12, thresholdDb: -40f);

        Assert.Equal(12, mask.Length);
        Assert.All(mask, value => Assert.InRange(value, 0f, 1f));
        Assert.Equal(0f, mask[0]);
        Assert.Equal(1f, mask[^1]);
        Assert.Equal(
            new[] { 0f, 0.1875f, 0.375f, 0.5625f, 0.75f, 0.9375f, 1f, 1f, 1f, 1f, 1f, 1f },
            mask,
            new FloatArrayComparer(1e-6f));
    }

    [Fact]
    public void MelProjection_DenormalizeUsesNaturalLogBaseByDefault() {
        var mel = MelProjection.Denormalize([-1f, 0f, 1f], "e");

        Assert.Equal(-12f, mel[0], 4);
        Assert.Equal(-5f, mel[1], 4);
        Assert.Equal(2f, mel[2], 4);
    }

    [Fact]
    public void MelProjection_DenormalizeConvertsToLog10WhenConfigured() {
        var mel = MelProjection.Denormalize([-1f, 0f, 1f], "log10");

        Assert.Equal(-12f * 0.434294f, mel[0], 4);
        Assert.Equal(-5f * 0.434294f, mel[1], 4);
        Assert.Equal(2f * 0.434294f, mel[2], 4);
    }

    [Fact]
    public void MelProjection_DenormalizeRejectsUnknownMelBase() {
        Assert.Throws<NotSupportedException>(() => MelProjection.Denormalize([0f], "weird"));
    }

    [Fact]
    public void AudioSlicer_SplitsLongMiddleSilenceIntoSeparateChunk() {
        const int sampleRate = 1000;
        var toneA = Enumerable.Range(0, 6000).Select(i => 0.1f * (float)Math.Sin(i * 0.1)).ToArray();
        var silence = new float[1000];
        var toneB = Enumerable.Range(0, 6000).Select(i => 0.1f * (float)Math.Sin(i * 0.15)).ToArray();
        var waveform = toneA.Concat(silence).Concat(toneB).ToArray();
        var slicer = new AudioSlicer(sampleRate);

        var chunks = slicer.Slice(waveform);

        Assert.Equal(3, chunks.Count);
        Assert.False(chunks[0].IsSilence);
        Assert.True(chunks[1].IsSilence);
        Assert.False(chunks[2].IsSilence);
        Assert.Equal(0, chunks[0].StartSample);
        Assert.Equal(waveform.Length, chunks[2].EndSample);
        Assert.InRange(chunks[1].StartSample, 5800, 6200);
        Assert.InRange(chunks[1].EndSample, 5800, 6200);
        Assert.Equal(chunks[1].StartSample, chunks[0].EndSample);
        Assert.Equal(chunks[1].EndSample, chunks[2].StartSample);
    }

    [Fact]
    public void ResolveOptions_ReadsModelAndExportMetadata() {
        var root = CreateTempDirectory();
        try {
            File.WriteAllText(Path.Combine(root, "config.yaml"), """
data:
  sampling_rate: 44100
  block_size: 512
  volume_smooth_size: 1024
  encoder: contentvec768l12tta2x
  encoder_sample_rate: 16000
  encoder_hop_size: 160
  f0_min: 65
  f0_max: 800
infer:
  infer_step: 50
  method: euler
model:
  t_start: 0.0
""");
            var onnxRoot = Path.Combine(root, "onnx");
            Directory.CreateDirectory(onnxRoot);
            File.WriteAllText(Path.Combine(onnxRoot, "svc.json"), """
{
  "sampling_rate": 44100,
  "block_size": 512,
  "encoder": "contentvec768l12tta2x",
  "n_spk": 3,
  "use_pitch_aug": true,
  "onnx": {
    "encoder": "encoder.onnx",
    "velocity": "velocity.onnx"
  }
}
""");

            var factory = new SvcPipelineFactory();
            var resolved = factory.ResolveOptions(new SvcRuntimeOptions { ModelRoot = onnxRoot });

            Assert.Equal(ContentEncoderKind.ContentVec768L12Tta2x, resolved.ContentEncoder);
            Assert.Equal(SvcExecutionProvider.Cpu, resolved.ExecutionProvider);
            Assert.Equal(0, resolved.ExecutionDeviceId);
            Assert.Equal(512, resolved.HopSize);
            Assert.Equal(1024, resolved.WinSize);
            Assert.Equal(50, resolved.ReflowSteps);
            Assert.Equal("euler", resolved.ReflowMethod);
            Assert.Equal(3, resolved.SpeakerCount);
            Assert.True(resolved.UsePitchAugmentation);
        } finally {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Runtime_CanBeCreatedFromResolvedAssetsWithoutLoadingModels() {
        var root = CreateTempDirectory();
        try {
            var onnxRoot = Path.Combine(root, "onnx");
            var dependenciesRoot = Path.Combine(root, "Dependencies");
            Directory.CreateDirectory(onnxRoot);
            Directory.CreateDirectory(dependenciesRoot);
            Directory.CreateDirectory(Path.Combine(dependenciesRoot, "rmvpe"));
            Directory.CreateDirectory(Path.Combine(dependenciesRoot, "contentvec"));
            var vocoderRoot = Path.Combine(dependenciesRoot, "pc_nsf_hifigan_test");
            Directory.CreateDirectory(vocoderRoot);

            File.WriteAllText(Path.Combine(root, "config.yaml"), """
data:
  sampling_rate: 44100
  block_size: 512
  volume_smooth_size: 1024
  encoder: contentvec768l12tta2x
infer:
  infer_step: 50
  method: euler
model:
  t_start: 0.0
""");
            File.WriteAllText(Path.Combine(onnxRoot, "svc.json"), """
{
  "sampling_rate": 44100,
  "block_size": 512,
  "n_spk": 3,
  "onnx": {
    "encoder": "encoder.onnx",
    "velocity": "velocity.onnx"
  }
}
""");
            File.WriteAllText(Path.Combine(onnxRoot, "encoder.onnx"), "stub");
            File.WriteAllText(Path.Combine(onnxRoot, "velocity.onnx"), "stub");
            File.WriteAllText(Path.Combine(dependenciesRoot, "rmvpe", "rmvpe.onnx"), "stub");
            File.WriteAllText(Path.Combine(dependenciesRoot, "contentvec", "contentvec_tta2x.onnx"), "stub");
            File.WriteAllText(Path.Combine(vocoderRoot, "model.onnx"), "stub");
            File.WriteAllText(Path.Combine(vocoderRoot, "vocoder.yaml"), """
model: model.onnx
sampling_rate: 44100
hop_size: 512
num_mels: 128
pitch_controllable: true
""");

            using var runtime = SvcRuntime.Create(new SvcRuntimeOptions {
                ModelRoot = onnxRoot,
                DependenciesRoot = dependenciesRoot,
            });

            Assert.Equal(3, runtime.Capabilities.SpeakerCount);
            Assert.True(runtime.Capabilities.SupportsSpeakerMix);
            Assert.False(runtime.Capabilities.SupportsFormantShift);
            Assert.True(runtime.Capabilities.SupportsNonZeroTStart);
            Assert.Equal(["euler", "rk4"], runtime.Capabilities.SupportedReflowMethods);

            var status = runtime.GetStatus();
            Assert.False(status.ContentEncoderLoaded);
            Assert.False(status.RmvpeLoaded);
            Assert.False(status.DDspEncoderLoaded);
            Assert.False(status.VelocityLoaded);
            Assert.False(status.VocoderLoaded);
        } finally {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ResolveOptions_PreservesExecutionProviderSelection() {
        var factory = new SvcPipelineFactory();

        var resolved = factory.ResolveOptions(new SvcRuntimeOptions {
            ModelRoot = "/tmp/nonexistent",
            ExecutionProvider = SvcExecutionProvider.Cuda,
            ExecutionDeviceId = 2,
        });

        Assert.Equal(SvcExecutionProvider.Cuda, resolved.ExecutionProvider);
        Assert.Equal(2, resolved.ExecutionDeviceId);
    }

    [Fact]
    public void Renderer_RejectsUnknownReflowMethodBeforeTouchingOnnx() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "bogus",
            SpeakerCount = 1,
        });
        var renderer = new SvcRenderer(pipeline);

        var ex = Assert.Throws<UnsupportedSvcFeatureException>(() => renderer.Render(new SvcInferenceRequest {
            Audio = [0.1f, 0.2f, 0.3f],
            SampleRate = 44100,
            SpeakerMix = [1f],
        }));

        Assert.Equal("reflow.method", ex.FeatureName);
    }

    [Fact]
    public void Renderer_AllowsRk4AndNonZeroTStartBeforeTouchingOnnx() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "rk4",
            ReflowTStart = 0.7f,
            SpeakerCount = 1,
            HopSize = 4,
        });
        var renderer = new SvcRenderer(pipeline);

        var info = renderer.Analyze(new SvcInferenceRequest {
            Audio = [0.1f, 0.2f, 0.3f, 0.4f],
            SampleRate = 44100,
        });

        Assert.Equal(50, info.ReflowSteps);
        Assert.Equal(44100, info.OutputSampleRate);
    }

    [Fact]
    public void Renderer_RejectsSpeakerMixForSingleSpeakerModels() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
        });
        var renderer = new SvcRenderer(pipeline);

        var ex = Assert.Throws<UnsupportedSvcFeatureException>(() => renderer.Render(new SvcInferenceRequest {
            Audio = [0.1f, 0.2f, 0.3f],
            SampleRate = 44100,
            SpeakerMix = [1f],
        }));

        Assert.Equal("speaker_mix", ex.FeatureName);
    }

    [Fact]
    public void Renderer_RejectsNonPositiveReflowSteps() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
        });
        var renderer = new SvcRenderer(pipeline);

        Assert.Throws<ArgumentOutOfRangeException>(() => renderer.Render(new SvcInferenceRequest {
            Audio = [0.1f, 0.2f, 0.3f],
            SampleRate = 44100,
            ReflowSteps = 0,
        }));
    }

    [Fact]
    public void Renderer_RejectsFormantAndRegisterShiftsExplicitly() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 2,
        });
        var renderer = new SvcRenderer(pipeline);

        var formant = Assert.Throws<UnsupportedSvcFeatureException>(() => renderer.Render(new SvcInferenceRequest {
            Audio = [0.1f, 0.2f, 0.3f],
            SampleRate = 44100,
            FormantShiftSemitones = 2,
        }));
        Assert.Equal("formant_shift_key", formant.FeatureName);

        var register = Assert.Throws<UnsupportedSvcFeatureException>(() => renderer.Render(new SvcInferenceRequest {
            Audio = [0.1f, 0.2f, 0.3f],
            SampleRate = 44100,
            VocalRegisterShiftSemitones = -2,
        }));
        Assert.Equal("vocal_register_shift_key", register.FeatureName);
    }

    [Fact]
    public void Renderer_AnalyzeReportsFrameAndSegmentMetadataWithoutLoadingOnnx() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 2,
            HopSize = 4,
        });
        var renderer = new SvcRenderer(pipeline);
        var audio = Enumerable.Repeat(0.1f, 24)
            .Concat(Enumerable.Repeat(0f, 8))
            .Concat(Enumerable.Repeat(0.1f, 24))
            .ToArray();

        var info = renderer.Analyze(new SvcInferenceRequest {
            Audio = audio,
            SampleRate = 1000,
            ReflowSteps = 32,
            KeyShiftSemitones = 3,
            SpeakerMix = [0.25f, 0.75f],
        });

        Assert.Equal(1000, info.InputSampleRate);
        Assert.Equal(44100, info.OutputSampleRate);
        Assert.Equal(audio.Length, info.InputSamples);
        Assert.Equal(2470, info.OutputSamples);
        Assert.Equal(618, info.TotalFrames);
        Assert.Equal(32, info.ReflowSteps);
        Assert.Equal(3, info.KeyShiftSemitones);
        Assert.True(info.UsedSpeakerMix);
        Assert.True(info.SegmentCount >= 1);
        var status = pipeline.GetStatus();
        Assert.False(status.ContentEncoderLoaded);
        Assert.False(status.RmvpeLoaded);
        Assert.False(status.DDspEncoderLoaded);
        Assert.False(status.VelocityLoaded);
        Assert.False(status.VocoderLoaded);
    }

    [Fact]
    public void Renderer_AnalyzeUsesOutputSampleRateForFrameAndSampleCounts() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
            HopSize = 4,
        });
        var renderer = new SvcRenderer(pipeline);
        var audio = Enumerable.Repeat(0.1f, 10).ToArray();

        var info = renderer.Analyze(new SvcInferenceRequest {
            Audio = audio,
            SampleRate = 22050,
        });

        Assert.Equal(22050, info.InputSampleRate);
        Assert.Equal(44100, info.OutputSampleRate);
        Assert.Equal(10, info.InputSamples);
        Assert.Equal(20, info.OutputSamples);
        Assert.Equal(6, info.TotalFrames);
        Assert.Equal(1, info.SegmentCount);
    }

    [Fact]
    public void Renderer_RenderEmptyInputReturnsRenderInfo() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
        });
        var renderer = new SvcRenderer(pipeline);

        var result = renderer.Render(new SvcInferenceRequest {
            Audio = [],
            SampleRate = 44100,
        });

        Assert.Empty(result.Audio);
        Assert.Equal(0, result.RenderInfo.InputSamples);
        Assert.Equal(0, result.RenderInfo.OutputSamples);
        Assert.Equal(0, result.RenderInfo.TotalFrames);
        Assert.Equal(0, result.RenderInfo.SegmentCount);
        Assert.Equal(44100, result.RenderInfo.OutputSampleRate);
    }

    [Fact]
    public void Renderer_PrepareConditioningEmptyInputReturnsNoSegmentsWithoutLoadingOnnx() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
        });
        var renderer = new SvcRenderer(pipeline);

        var conditioning = renderer.PrepareConditioning(new SvcInferenceRequest {
            Audio = [],
            SampleRate = 44100,
        });

        Assert.Empty(conditioning.Segments);
        Assert.Empty(conditioning.VolumeMask);
        Assert.Equal(0, conditioning.TotalFrames);
        var status = pipeline.GetStatus();
        Assert.False(status.ContentEncoderLoaded);
        Assert.False(status.RmvpeLoaded);
        Assert.False(status.DDspEncoderLoaded);
        Assert.False(status.VelocityLoaded);
        Assert.False(status.VocoderLoaded);
    }

    [Fact]
    public void Renderer_RenderPreparedEmptyConditioningReturnsSilentResult() {
        using var pipeline = CreatePipeline(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
        });
        var renderer = new SvcRenderer(pipeline);

        var result = renderer.Render(new SvcPreparedConditioning {
            InputSampleRate = 44100,
            OutputSampleRate = 44100,
            HopSize = 512,
            ReflowSteps = 50,
            KeyShiftSemitones = 0,
            SilenceThresholdDb = -60f,
            TotalFrames = 0,
            VolumeMask = [],
            SpeakerMix = null,
            Segments = [],
            RenderInfo = new SvcRenderInfo(
                InputSampleRate: 44100,
                OutputSampleRate: 44100,
                InputSamples: 0,
                OutputSamples: 0,
                TotalFrames: 0,
                SegmentCount: 0,
                ReflowSteps: 50,
                KeyShiftSemitones: 0,
                SilenceThresholdDb: -60f,
                UsedSpeakerMix: false),
        });

        Assert.Empty(result.Audio);
        Assert.Equal(0, result.RenderInfo.OutputSamples);
        Assert.Equal(0, result.RenderInfo.SegmentCount);
    }

    [Fact]
    public void HostRenderRequest_MapsToInferenceRequest() {
        var hostRequest = new SvcHostRenderRequest {
            RequestId = "phrase-1",
            CacheKey = "cache-1",
            Audio = [0.1f, 0.2f],
            SampleRate = 44100,
            Seed = 123,
            ReflowSteps = 20,
            SilenceThresholdDb = -55f,
            KeyShiftSemitones = 2,
            FormantShiftSemitones = 1,
            VocalRegisterShiftSemitones = -1,
            SpeakerMix = [0.25f, 0.75f],
        };

        var inference = hostRequest.ToInferenceRequest();

        Assert.Equal(hostRequest.Audio, inference.Audio);
        Assert.Equal(44100, inference.SampleRate);
        Assert.Equal(123, inference.Seed);
        Assert.Equal(20, inference.ReflowSteps);
        Assert.Equal(-55f, inference.SilenceThresholdDb);
        Assert.Equal(2, inference.KeyShiftSemitones);
        Assert.Equal(1, inference.FormantShiftSemitones);
        Assert.Equal(-1, inference.VocalRegisterShiftSemitones);
        Assert.Equal(hostRequest.SpeakerMix, inference.SpeakerMix);
    }

    [Fact]
    public void HostRenderService_PreparePreservesRequestIdentityAndAvoidsModelLoadForEmptyInput() {
        var runtime = CreateRuntime(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
        });
        var service = new SvcHostRenderService(runtime);

        var prepared = service.Prepare(new SvcHostRenderRequest {
            RequestId = "phrase-42",
            CacheKey = "trackA:phrase-42",
            Audio = [],
            SampleRate = 44100,
            Seed = 7,
        });

        Assert.Equal("phrase-42", prepared.RequestId);
        Assert.Equal("trackA:phrase-42", prepared.CacheKey);
        Assert.Equal(7, prepared.Conditioning.Seed);
        Assert.Empty(prepared.Conditioning.Segments);

        var status = runtime.GetStatus();
        Assert.False(status.ContentEncoderLoaded);
        Assert.False(status.RmvpeLoaded);
        Assert.False(status.DDspEncoderLoaded);
        Assert.False(status.VelocityLoaded);
        Assert.False(status.VocoderLoaded);
    }

    [Fact]
    public void HostRenderService_RenderPreparedPreservesRequestIdentity() {
        var runtime = CreateRuntime(new SvcRuntimeOptions {
            ReflowMethod = "euler",
            SpeakerCount = 1,
        });
        var service = new SvcHostRenderService(runtime);

        var result = service.Render(new SvcHostPreparedRender {
            RequestId = "phrase-99",
            CacheKey = "cache-99",
            Conditioning = new SvcPreparedConditioning {
                InputSampleRate = 44100,
                OutputSampleRate = 44100,
                HopSize = 512,
                Seed = 11,
                ReflowSteps = 50,
                KeyShiftSemitones = 0,
                SilenceThresholdDb = -60f,
                TotalFrames = 0,
                VolumeMask = [],
                SpeakerMix = null,
                Segments = [],
                RenderInfo = new SvcRenderInfo(
                    InputSampleRate: 44100,
                    OutputSampleRate: 44100,
                    InputSamples: 0,
                    OutputSamples: 0,
                    TotalFrames: 0,
                    SegmentCount: 0,
                    ReflowSteps: 50,
                    KeyShiftSemitones: 0,
                    SilenceThresholdDb: -60f,
                    UsedSpeakerMix: false),
            },
        });

        Assert.Equal("phrase-99", result.RequestId);
        Assert.Equal("cache-99", result.CacheKey);
        Assert.Empty(result.Audio);
        Assert.Equal(0, result.RenderInfo.OutputSamples);
    }

    private static string CreateTempDirectory() {
        var path = Path.Combine(Path.GetTempPath(), "DdspSvc.OnnxRuntimeTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private static SvcInferencePipeline CreatePipeline(SvcRuntimeOptions options) {
        return new SvcInferencePipeline(
            options,
            new SvcModelPaths("encoder.onnx", "velocity.onnx", "vocoder.onnx", "rmvpe.onnx", "contentvec.onnx"),
            new VocoderConfig {
                Model = "vocoder.onnx",
                SampleRate = 44100,
                HopSize = 512,
                NumMelBins = 128,
                PitchControllable = true,
            });
    }

    private static SvcRuntime CreateRuntime(SvcRuntimeOptions options) {
        return SvcRuntime.Create(CreatePipeline(options));
    }

    private sealed class FloatArrayComparer(float tolerance) : IEqualityComparer<float> {
        public bool Equals(float x, float y) => Math.Abs(x - y) <= tolerance;

        public int GetHashCode(float obj) => 0;
    }
}

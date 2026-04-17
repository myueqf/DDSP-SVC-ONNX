using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DdspSvc.OnnxRuntime.Audio;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Onnx;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class ContentVecEncoder : IDisposable {
    private const int MinSamples = 32000;
    private readonly InferenceSession session;
    private readonly ContentEncoderKind encoderKind;
    private readonly bool usesPreExportedTtaModel;
    private readonly int encoderSampleRate;
    private readonly int encoderHopSize;
    private readonly string inputName;
    private readonly string outputName;
    private bool disposed;

    public ContentVecEncoder(
        string modelPath,
        ContentEncoderKind encoderKind,
        int encoderSampleRate,
        int encoderHopSize,
        OnnxSessionFactory? sessionFactory = null) {
        if (!File.Exists(modelPath)) {
            throw new FileNotFoundException("ContentVec model not found.", modelPath);
        }
        this.encoderKind = encoderKind;
        this.encoderSampleRate = encoderSampleRate;
        this.encoderHopSize = encoderHopSize;
        usesPreExportedTtaModel = Path.GetFileName(modelPath).Contains("tta2x", StringComparison.OrdinalIgnoreCase);
        session = (sessionFactory ?? new OnnxSessionFactory()).Create(modelPath, OnnxRunnerChoice.Cpu);
        inputName = session.InputNames.First();
        outputName = session.OutputNames.First();
    }

    public ContentUnitsResult Encode(ReadOnlySpan<float> monoSamples, int sourceSampleRate, int targetHopSize) {
        ObjectDisposedException.ThrowIf(disposed, this);

        var resampled = LinearResampler.Resample(monoSamples, sourceSampleRate, encoderSampleRate);
        if (encoderKind == ContentEncoderKind.ContentVec768L12Tta2x && !usesPreExportedTtaModel) {
            return AlignUnits(EncodeTta2x(resampled), monoSamples.Length, sourceSampleRate, targetHopSize);
        }

        return AlignUnits(RunOnce(resampled), monoSamples.Length, sourceSampleRate, targetHopSize);
    }

    public void Dispose() {
        if (disposed) {
            return;
        }
        session.Dispose();
        disposed = true;
    }

    private ContentUnitsResult EncodeTta2x(float[] audio) {
        var baseUnits = RunOnce(audio);

        var shiftedAudio = new float[audio.Length + 160];
        Array.Copy(audio, 0, shiftedAudio, 160, audio.Length);
        var shiftedUnits = RunOnce(shiftedAudio);

        var n = shiftedUnits.Frames - baseUnits.Frames;
        var baseFrameCount = baseUnits.Frames + Math.Max(0, n);
        var basePadded = new float[baseFrameCount * baseUnits.Channels];
        Array.Copy(baseUnits.Units, basePadded, baseUnits.Units.Length);
        if (n > 0 && baseUnits.Frames > 0) {
            var lastFrameOffset = (baseUnits.Frames - 1) * baseUnits.Channels;
            for (var frame = baseUnits.Frames; frame < baseFrameCount; frame++) {
                Array.Copy(baseUnits.Units, lastFrameOffset, basePadded, frame * baseUnits.Channels, baseUnits.Channels);
            }
        }

        var mergedFrames = Math.Min(baseFrameCount, shiftedUnits.Frames) * 2;
        var merged = new float[mergedFrames * baseUnits.Channels];
        var cursor = 0;
        for (var frame = 0; frame < Math.Min(baseFrameCount, shiftedUnits.Frames); frame++) {
            Array.Copy(shiftedUnits.Units, frame * shiftedUnits.Channels, merged, cursor * shiftedUnits.Channels, shiftedUnits.Channels);
            cursor++;
            Array.Copy(basePadded, frame * baseUnits.Channels, merged, cursor * baseUnits.Channels, baseUnits.Channels);
            cursor++;
        }

        var startFrame = 1;
        var endFrameExclusive = cursor - Math.Max(0, n);
        var finalFrames = Math.Max(0, endFrameExclusive - startFrame);
        var finalUnits = new float[finalFrames * baseUnits.Channels];
        if (finalFrames > 0) {
            Array.Copy(merged, startFrame * baseUnits.Channels, finalUnits, 0, finalUnits.Length);
        }

        return new ContentUnitsResult {
            Frames = finalFrames,
            Channels = baseUnits.Channels,
            Units = finalUnits,
        };
    }

    private ContentUnitsResult RunOnce(float[] audioSamples) {
        var paddedLength = Math.Max(audioSamples.Length, MinSamples);
        var audio = new DenseTensor<float>(new[] { 1, paddedLength });
        for (var i = 0; i < paddedLength; i++) {
            var sample = i < audioSamples.Length ? audioSamples[i] : 0f;
            audio[0, i] = Math.Clamp(sample, -1f, 1f);
        }

        using var outputs = session.Run([NamedOnnxValue.CreateFromTensor(inputName, audio)]);
        var units = outputs.First(output => output.Name == outputName).AsTensor<float>();
        return new ContentUnitsResult {
            Frames = units.Dimensions[1],
            Channels = units.Dimensions[2],
            Units = units.ToArray(),
        };
    }

    private ContentUnitsResult AlignUnits(ContentUnitsResult units, int originalSampleCount, int sourceSampleRate, int targetHopSize) {
        var targetFrames = originalSampleCount / targetHopSize + 1;
        var ratio = (targetHopSize / (double)sourceSampleRate) / (encoderHopSize / (double)encoderSampleRate);
        var aligned = new float[targetFrames * units.Channels];
        for (var frame = 0; frame < targetFrames; frame++) {
            var sourceFrame = Math.Min((int)Math.Round(ratio * frame), units.Frames - 1);
            Array.Copy(units.Units, sourceFrame * units.Channels, aligned, frame * units.Channels, units.Channels);
        }
        return new ContentUnitsResult {
            Frames = targetFrames,
            Channels = units.Channels,
            Units = aligned,
        };
    }
}

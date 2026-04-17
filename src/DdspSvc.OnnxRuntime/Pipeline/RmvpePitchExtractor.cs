using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DdspSvc.OnnxRuntime.Audio;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Onnx;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class RmvpePitchExtractor : IDisposable {
    private const int DefaultSampleRate = 16000;
    private const int DefaultHopLength = 160;
    private const float DefaultThreshold = 0.03f;

    private readonly InferenceSession session;
    private readonly string waveformInputName;
    private readonly string thresholdInputName;
    private readonly string f0OutputName;
    private readonly string uvOutputName;
    private readonly RunOptions runOptions = new();
    private bool disposed;

    public RmvpePitchExtractor(string modelPath, OnnxSessionFactory? sessionFactory = null) {
        if (!File.Exists(modelPath)) {
            throw new FileNotFoundException("RMVPE model not found.", modelPath);
        }
        session = (sessionFactory ?? new OnnxSessionFactory()).Create(modelPath, OnnxRunnerChoice.Cpu);
        waveformInputName = ResolveWaveformInputName(session);
        thresholdInputName = ResolveThresholdInputName(session);
        f0OutputName = ResolveF0OutputName(session);
        uvOutputName = ResolveUvOutputName(session);
    }

    public RmvpeResult Infer(ReadOnlySpan<float> monoSamples, int sourceSampleRate, int targetHopSize, float minF0Hz = 65f) {
        ObjectDisposedException.ThrowIf(disposed, this);

        var resampled = LinearResampler.Resample(monoSamples, sourceSampleRate, DefaultSampleRate);
        var waveform = new DenseTensor<float>(new[] { 1, resampled.Length });
        for (var i = 0; i < resampled.Length; i++) {
            waveform[0, i] = Math.Clamp(resampled[i], -1f, 1f);
        }
        var threshold = new DenseTensor<float>(new[] { DefaultThreshold }, []);

        using var outputs = session.Run(
            [
                NamedOnnxValue.CreateFromTensor(waveformInputName, waveform),
                NamedOnnxValue.CreateFromTensor(thresholdInputName, threshold),
            ],
            session.OutputNames,
            runOptions);

        var rawF0 = outputs.First(output => output.Name == f0OutputName).AsTensor<float>().ToArray();
        var rawUv = outputs.First(output => output.Name == uvOutputName).AsTensor<bool>().ToArray();
        if (rawF0.Length != rawUv.Length) {
            throw new InvalidDataException($"Unexpected RMVPE output sizes: f0={rawF0.Length}, uv={rawUv.Length}");
        }

        var interpolatedF0 = InterpolateUnvoicedForResample(rawF0, rawUv);
        var nFrames = monoSamples.Length / targetHopSize + 1;
        var targetTimeStep = targetHopSize / (double)sourceSampleRate;
        var f0 = new float[nFrames];
        var uv = new bool[nFrames];
        for (var frame = 0; frame < nFrames; frame++) {
            var targetTime = frame * targetTimeStep;
            f0[frame] = InterpolateAt(interpolatedF0, 0.01, targetTime);
            uv[frame] = InterpolateAt(rawUv, 0.01, targetTime) > 0.5f;
            if (uv[frame]) {
                f0[frame] = 0f;
            }
        }

        // Match uv_interp=True in the original pipeline.
        if (f0.Any(value => value > 0f)) {
            var conditioned = PitchConditioning.InterpolateUnvoiced(f0, minF0Hz);
            Array.Copy(conditioned, f0, conditioned.Length);
        }

        return new RmvpeResult {
            TimeStepSeconds = targetTimeStep,
            F0Hz = f0,
            Unvoiced = uv,
            MidiPitch = ConvertToInterpolatedMidiPitch(f0, uv),
        };
    }

    public void Interrupt() {
        if (!disposed) {
            runOptions.Terminate = true;
        }
    }

    public void Dispose() {
        if (disposed) {
            return;
        }
        runOptions.Dispose();
        session.Dispose();
        disposed = true;
    }

    private static float[] ConvertToInterpolatedMidiPitch(float[] f0, bool[] uv) {
        var midi = new float[f0.Length];
        for (var i = 0; i < midi.Length; ++i) {
            var voiced = !uv[i] && f0[i] > 0;
            midi[i] = voiced
                ? (float)(69.0 + 12.0 * Math.Log2(f0[i] / 440.0))
                : float.NaN;
        }
        InterpolateMidiPitch(midi);
        return midi;
    }

    private static void InterpolateMidiPitch(float[] midi) {
        var firstVoiced = -1;
        for (var i = 0; i < midi.Length; ++i) {
            if (!float.IsNaN(midi[i])) {
                firstVoiced = i;
                break;
            }
        }
        if (firstVoiced < 0) {
            return;
        }
        for (var i = 0; i < firstVoiced; ++i) {
            midi[i] = midi[firstVoiced];
        }
        var previousVoiced = firstVoiced;
        var index = firstVoiced + 1;
        while (index < midi.Length) {
            if (!float.IsNaN(midi[index])) {
                previousVoiced = index;
                ++index;
                continue;
            }
            var gapStart = index;
            while (index < midi.Length && float.IsNaN(midi[index])) {
                ++index;
            }
            if (index < midi.Length) {
                var left = midi[previousVoiced];
                var right = midi[index];
                var gapLength = index - previousVoiced;
                for (var i = 1; i < gapLength; ++i) {
                    var ratio = (float)i / gapLength;
                    midi[previousVoiced + i] = left + (right - left) * ratio;
                }
                previousVoiced = index;
            } else {
                for (var i = gapStart; i < midi.Length; ++i) {
                    midi[i] = midi[previousVoiced];
                }
            }
        }
    }

    private static float[] InterpolateUnvoicedForResample(float[] f0, bool[] uv) {
        var result = (float[])f0.Clone();
        var voiced = Enumerable.Range(0, result.Length)
            .Where(i => !uv[i] && result[i] > 0f)
            .ToArray();
        if (voiced.Length == 0) {
            return result;
        }

        for (var i = 0; i < result.Length; i++) {
            if (!uv[i] && result[i] > 0f) {
                continue;
            }
            var right = Array.FindIndex(voiced, idx => idx >= i);
            if (right < 0) {
                result[i] = result[voiced[^1]];
                continue;
            }
            if (voiced[right] == i || right == 0) {
                result[i] = result[voiced[right]];
                continue;
            }

            var leftIndex = voiced[right - 1];
            var rightIndex = voiced[right];
            var ratio = (float)(i - leftIndex) / (rightIndex - leftIndex);
            result[i] = result[leftIndex] + (result[rightIndex] - result[leftIndex]) * ratio;
        }

        return result;
    }

    private static float InterpolateAt(float[] values, double stepSeconds, double targetTime) {
        if (values.Length == 0) {
            return 0f;
        }
        var position = targetTime / stepSeconds;
        var left = Math.Clamp((int)Math.Floor(position), 0, values.Length - 1);
        var right = Math.Clamp(left + 1, 0, values.Length - 1);
        var frac = (float)(position - left);
        return values[left] + (values[right] - values[left]) * frac;
    }

    private static float InterpolateAt(bool[] values, double stepSeconds, double targetTime) {
        if (values.Length == 0) {
            return 0f;
        }
        var position = targetTime / stepSeconds;
        var left = Math.Clamp((int)Math.Floor(position), 0, values.Length - 1);
        var right = Math.Clamp(left + 1, 0, values.Length - 1);
        var frac = (float)(position - left);
        var leftValue = values[left] ? 1f : 0f;
        var rightValue = values[right] ? 1f : 0f;
        return leftValue + (rightValue - leftValue) * frac;
    }

    private static string ResolveWaveformInputName(InferenceSession session) {
        return session.InputNames.FirstOrDefault(name =>
                string.Equals(name, "waveform", StringComparison.OrdinalIgnoreCase))
            ?? session.InputNames.First();
    }

    private static string ResolveThresholdInputName(InferenceSession session) {
        return session.InputNames.FirstOrDefault(name =>
                string.Equals(name, "threshold", StringComparison.OrdinalIgnoreCase))
            ?? session.InputNames.ElementAtOrDefault(1)
            ?? throw new InvalidDataException("RMVPE model must expose a threshold input.");
    }

    private static string ResolveF0OutputName(InferenceSession session) {
        return session.OutputNames.FirstOrDefault(name =>
                string.Equals(name, "f0", StringComparison.OrdinalIgnoreCase))
            ?? session.OutputNames.First();
    }

    private static string ResolveUvOutputName(InferenceSession session) {
        return session.OutputNames.FirstOrDefault(name =>
                string.Equals(name, "uv", StringComparison.OrdinalIgnoreCase))
            ?? session.OutputNames.ElementAtOrDefault(1)
            ?? throw new InvalidDataException("RMVPE model must expose a uv output.");
    }
}

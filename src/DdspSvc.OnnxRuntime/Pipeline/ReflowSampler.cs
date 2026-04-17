using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Onnx;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class ReflowSampler : IDisposable {
    private readonly InferenceSession session;
    private readonly string xInputName;
    private readonly string tInputName;
    private readonly string condInputName;
    private readonly string outputName;
    private bool disposed;

    public ReflowSampler(string modelPath, OnnxSessionFactory? sessionFactory = null) {
        if (!File.Exists(modelPath)) {
            throw new FileNotFoundException("Reflow model not found.", modelPath);
        }
        session = (sessionFactory ?? new OnnxSessionFactory()).Create(modelPath);
        xInputName = session.InputNames[0];
        tInputName = session.InputNames[1];
        condInputName = session.InputNames[2];
        outputName = session.OutputNames[0];
    }

    public float[] Sample(
        DdspEncoderResult ddsp,
        int steps = 20,
        string method = "euler",
        float tStart = 0f,
        int? seed = null) {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (steps <= 0) {
            throw new ArgumentOutOfRangeException(nameof(steps));
        }
        if (tStart is < 0f or >= 1f) {
            throw new ArgumentOutOfRangeException(nameof(tStart), "tStart must be in [0, 1).");
        }

        var rng = seed.HasValue ? new Random(unchecked(seed.Value ^ 0x4f1bbcdc)) : Random.Shared;
        var x = new float[ddsp.X.Length];
        for (var i = 0; i < x.Length; i++) {
            var noise = NextGaussian(rng);
            x[i] = tStart > 0f
                ? tStart * ddsp.X[i] + (1f - tStart) * noise
                : noise;
        }
        var dt = (1f - tStart) / steps;
        var t = tStart;
        for (var i = 0; i < steps; i++) {
            if (string.Equals(method, "euler", StringComparison.OrdinalIgnoreCase)) {
                var velocity = RunVelocity(x, t, ddsp);
                for (var j = 0; j < x.Length; j++) {
                    x[j] += velocity[j] * dt;
                }
            } else if (string.Equals(method, "rk4", StringComparison.OrdinalIgnoreCase)) {
                var k1 = RunVelocity(x, t, ddsp);
                var x2 = AddScaled(x, k1, 0.5f * dt);
                var k2 = RunVelocity(x2, t + 0.5f * dt, ddsp);
                var x3 = AddScaled(x, k2, 0.5f * dt);
                var k3 = RunVelocity(x3, t + 0.5f * dt, ddsp);
                var x4 = AddScaled(x, k3, dt);
                var k4 = RunVelocity(x4, t + dt, ddsp);
                for (var j = 0; j < x.Length; j++) {
                    x[j] += (k1[j] + 2f * k2[j] + 2f * k3[j] + k4[j]) * dt / 6f;
                }
            } else {
                throw new NotSupportedException($"Unsupported reflow method '{method}'.");
            }
            t += dt;
        }
        return x;
    }

    public void Dispose() {
        if (disposed) {
            return;
        }
        session.Dispose();
        disposed = true;
    }

    private static float NextGaussian(Random rng) {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    private float[] RunVelocity(float[] x, float t, DdspEncoderResult ddsp) {
        var tTensor = new DenseTensor<long>(new[] { (long)Math.Round(t * 1000f) }, [1]);
        var xTensor = new DenseTensor<float>(x, [1, 1, ddsp.MelBins, ddsp.Frames]);
        var condTensor = new DenseTensor<float>(ddsp.Cond, [1, ddsp.MelBins, ddsp.Frames]);
        using var outputs = session.Run(
            [
                NamedOnnxValue.CreateFromTensor(xInputName, xTensor),
                NamedOnnxValue.CreateFromTensor(tInputName, tTensor),
                NamedOnnxValue.CreateFromTensor(condInputName, condTensor),
            ]);
        return outputs.First(output => output.Name == outputName).AsTensor<float>().ToArray();
    }

    private static float[] AddScaled(float[] x, float[] delta, float scale) {
        var result = new float[x.Length];
        for (var i = 0; i < x.Length; i++) {
            result[i] = x[i] + delta[i] * scale;
        }
        return result;
    }
}

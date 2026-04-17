using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Onnx;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class DdspEncoder : IDisposable {
    private readonly InferenceSession session;
    private readonly string hubertInputName;
    private readonly string mel2phInputName;
    private readonly string f0InputName;
    private readonly string volumeInputName;
    private readonly string? speakerMixInputName;
    private readonly string randnInputName;
    private readonly string xOutputName;
    private readonly string condOutputName;
    private readonly int blockSize;
    private bool disposed;

    public DdspEncoder(string modelPath, int blockSize, OnnxSessionFactory? sessionFactory = null) {
        if (!File.Exists(modelPath)) {
            throw new FileNotFoundException("DDSP encoder model not found.", modelPath);
        }
        this.blockSize = blockSize;
        session = (sessionFactory ?? new OnnxSessionFactory()).Create(modelPath);
        hubertInputName = ResolveInputName("hubert", 0);
        mel2phInputName = ResolveInputName("mel2ph", 1);
        f0InputName = ResolveInputName("f0", 2);
        volumeInputName = ResolveInputName("volume", 3);
        speakerMixInputName = session.InputNames.FirstOrDefault(name =>
            string.Equals(name, "spk_mix", StringComparison.OrdinalIgnoreCase));
        randnInputName = ResolveInputName("randn", 4);
        xOutputName = session.OutputNames[0];
        condOutputName = session.OutputNames[1];
    }

    public DdspEncoderResult Infer(
        ContentUnitsResult units,
        float[] f0,
        float[] volume,
        int? seed = null,
        float[]? speakerMix = null) {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (f0.Length != units.Frames || volume.Length != units.Frames) {
            throw new ArgumentException("f0 and volume lengths must match unit frame count.");
        }

        var hubert = new DenseTensor<float>(units.Units, [1, units.Frames, units.Channels]);
        var mel2phData = Enumerable.Range(0, units.Frames).Select(i => (long)i).ToArray();
        var mel2ph = new DenseTensor<long>(mel2phData, [1, units.Frames]);
        var f0Tensor = new DenseTensor<float>(f0, [1, units.Frames]);
        var volumeTensor = new DenseTensor<float>(volume, [1, units.Frames]);

        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var randn = new float[units.Frames * blockSize];
        for (var i = 0; i < randn.Length; i++) {
            randn[i] = NextGaussian(rng);
        }
        var randnTensor = new DenseTensor<float>(randn, [1, randn.Length]);

        var inputs = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor(hubertInputName, hubert),
            NamedOnnxValue.CreateFromTensor(mel2phInputName, mel2ph),
            NamedOnnxValue.CreateFromTensor(f0InputName, f0Tensor),
            NamedOnnxValue.CreateFromTensor(volumeInputName, volumeTensor),
        };
        if (speakerMixInputName is not null) {
            speakerMix ??= [1f];
            var speakerMixTensor = BroadcastSpeakerMix(speakerMix, units.Frames);
            inputs.Add(NamedOnnxValue.CreateFromTensor(speakerMixInputName, speakerMixTensor));
        }
        inputs.Add(NamedOnnxValue.CreateFromTensor(randnInputName, randnTensor));

        using (var outputs = session.Run(inputs)) {
            var x = outputs.First(output => output.Name == xOutputName).AsTensor<float>();
            var cond = outputs.First(output => output.Name == condOutputName).AsTensor<float>();
            return new DdspEncoderResult {
                Frames = cond.Dimensions[2],
                MelBins = cond.Dimensions[1],
                X = x.ToArray(),
                Cond = cond.ToArray(),
            };
        }
    }

    private static DenseTensor<float> BroadcastSpeakerMix(float[] speakerMix, int frames) {
        if (speakerMix.Length == 0) {
            throw new ArgumentException("Speaker mix cannot be empty.", nameof(speakerMix));
        }
        var data = new float[frames * speakerMix.Length];
        for (var frame = 0; frame < frames; frame++) {
            Array.Copy(speakerMix, 0, data, frame * speakerMix.Length, speakerMix.Length);
        }
        return new DenseTensor<float>(data, [1, frames, speakerMix.Length]);
    }

    public void Dispose() {
        if (disposed) {
            return;
        }
        session.Dispose();
        disposed = true;
    }

    private string ResolveInputName(string preferred, int fallbackIndex) {
        return session.InputNames.FirstOrDefault(name =>
                string.Equals(name, preferred, StringComparison.OrdinalIgnoreCase))
            ?? session.InputNames.ElementAtOrDefault(fallbackIndex)
            ?? throw new InvalidDataException($"Unable to resolve input '{preferred}'.");
    }

    private static float NextGaussian(Random rng) {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}

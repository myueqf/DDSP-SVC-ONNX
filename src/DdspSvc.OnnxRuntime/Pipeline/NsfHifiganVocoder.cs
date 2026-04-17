using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Onnx;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class NsfHifiganVocoder : IDisposable {
    private readonly InferenceSession session;
    private readonly VocoderConfig config;
    private readonly string melInputName;
    private readonly string f0InputName;
    private readonly string outputName;
    private bool disposed;

    public NsfHifiganVocoder(
        string modelPath,
        VocoderConfig config,
        OnnxSessionFactory? sessionFactory = null) {
        if (!File.Exists(modelPath)) {
            throw new FileNotFoundException("Vocoder model not found.", modelPath);
        }
        this.config = config;
        session = (sessionFactory ?? new OnnxSessionFactory()).Create(modelPath);
        melInputName = ResolveInputName(session, "mel", 0);
        f0InputName = ResolveInputName(session, "f0", 1);
        outputName = session.OutputNames.First();
    }

    public VocoderResult Infer(float[] mel, int melFrames, int melBins, float[] f0) {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (melFrames <= 0) {
            throw new ArgumentOutOfRangeException(nameof(melFrames));
        }
        if (melBins <= 0) {
            throw new ArgumentOutOfRangeException(nameof(melBins));
        }
        if (mel.Length != melFrames * melBins) {
            throw new ArgumentException("Mel tensor size does not match melFrames * melBins.", nameof(mel));
        }
        if (f0.Length != melFrames) {
            throw new ArgumentException("F0 frame count must equal mel frame count.", nameof(f0));
        }

        var melTensor = new DenseTensor<float>(mel, [1, melFrames, melBins]);
        var f0Tensor = new DenseTensor<float>(f0, [1, melFrames]);

        using var outputs = session.Run(
            [
                NamedOnnxValue.CreateFromTensor(melInputName, melTensor),
                NamedOnnxValue.CreateFromTensor(f0InputName, f0Tensor),
            ]);

        var audio = outputs.First(output => output.Name == outputName).AsTensor<float>().ToArray();
        return new VocoderResult {
            SampleRate = config.SampleRate,
            HopSize = config.HopSize,
            Audio = audio,
        };
    }

    public void Dispose() {
        if (disposed) {
            return;
        }
        session.Dispose();
        disposed = true;
    }

    private static string ResolveInputName(InferenceSession session, string preferred, int fallbackIndex) {
        return session.InputNames.FirstOrDefault(name =>
                string.Equals(name, preferred, StringComparison.OrdinalIgnoreCase))
            ?? session.InputNames.ElementAtOrDefault(fallbackIndex)
            ?? throw new InvalidDataException($"Unable to resolve input '{preferred}'.");
    }
}

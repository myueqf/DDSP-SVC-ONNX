namespace DdspSvc.OnnxRuntime.Models;

public sealed class VocoderResult {
    public int SampleRate { get; init; }
    public int HopSize { get; init; }
    public float[] Audio { get; init; } = [];
}

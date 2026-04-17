namespace DdspSvc.OnnxRuntime.Models;

public sealed class DdspEncoderResult {
    public int Frames { get; init; }
    public int MelBins { get; init; }
    public float[] X { get; init; } = [];
    public float[] Cond { get; init; } = [];
}

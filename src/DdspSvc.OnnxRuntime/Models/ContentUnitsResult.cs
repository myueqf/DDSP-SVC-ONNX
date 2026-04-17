namespace DdspSvc.OnnxRuntime.Models;

public sealed class ContentUnitsResult {
    public int Frames { get; init; }
    public int Channels { get; init; }
    public float[] Units { get; init; } = [];
}

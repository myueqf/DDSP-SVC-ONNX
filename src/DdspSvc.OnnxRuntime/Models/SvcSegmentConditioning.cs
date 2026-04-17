namespace DdspSvc.OnnxRuntime.Models;

public sealed class SvcSegmentConditioning {
    public required int StartFrame { get; init; }
    public required int EndFrame { get; init; }
    public required int StartSample { get; init; }
    public required int EndSample { get; init; }
    public required ContentUnitsResult Units { get; init; }
    public required float[] F0Hz { get; init; }
    public required float[] Volume { get; init; }
}

using DdspSvc.OnnxRuntime.Models;

namespace DdspSvc.OnnxRuntime.Hosting;

public sealed class SvcHostRenderResult {
    public string? RequestId { get; init; }
    public string? CacheKey { get; init; }
    public required float[] Audio { get; init; }
    public required int SampleRate { get; init; }
    public required SvcRenderInfo RenderInfo { get; init; }
}

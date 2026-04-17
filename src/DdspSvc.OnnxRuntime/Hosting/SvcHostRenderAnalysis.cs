using DdspSvc.OnnxRuntime.Models;

namespace DdspSvc.OnnxRuntime.Hosting;

public sealed class SvcHostRenderAnalysis {
    public string? RequestId { get; init; }
    public string? CacheKey { get; init; }
    public required SvcRenderInfo RenderInfo { get; init; }
}

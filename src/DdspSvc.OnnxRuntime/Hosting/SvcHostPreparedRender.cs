using DdspSvc.OnnxRuntime.Models;

namespace DdspSvc.OnnxRuntime.Hosting;

public sealed class SvcHostPreparedRender {
    public string? RequestId { get; init; }
    public string? CacheKey { get; init; }
    public required SvcPreparedConditioning Conditioning { get; init; }
}

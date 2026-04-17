using DdspSvc.OnnxRuntime.Models;

namespace DdspSvc.OnnxRuntime.Hosting;

public sealed class SvcHostRenderService {
    private readonly SvcRuntime runtime;

    public SvcHostRenderService(SvcRuntime runtime) {
        this.runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
    }

    public void Validate(SvcHostRenderRequest request) {
        ArgumentNullException.ThrowIfNull(request);
        runtime.ValidateRequest(request.ToInferenceRequest());
    }

    public SvcHostRenderAnalysis Analyze(SvcHostRenderRequest request) {
        ArgumentNullException.ThrowIfNull(request);
        return new SvcHostRenderAnalysis {
            RequestId = request.RequestId,
            CacheKey = request.CacheKey,
            RenderInfo = runtime.AnalyzeRequest(request.ToInferenceRequest()),
        };
    }

    public SvcHostPreparedRender Prepare(SvcHostRenderRequest request) {
        ArgumentNullException.ThrowIfNull(request);
        return new SvcHostPreparedRender {
            RequestId = request.RequestId,
            CacheKey = request.CacheKey,
            Conditioning = runtime.PrepareConditioning(request.ToInferenceRequest()),
        };
    }

    public SvcHostRenderResult Render(SvcHostRenderRequest request) {
        ArgumentNullException.ThrowIfNull(request);
        return FromInferenceResult(request.RequestId, request.CacheKey, runtime.Render(request.ToInferenceRequest()));
    }

    public SvcHostRenderResult Render(SvcHostPreparedRender prepared) {
        ArgumentNullException.ThrowIfNull(prepared);
        var result = runtime.RenderPrepared(prepared.Conditioning);
        return FromInferenceResult(prepared.RequestId, prepared.CacheKey, result);
    }

    private static SvcHostRenderResult FromInferenceResult(string? requestId, string? cacheKey, SvcInferenceResult result) {
        return new SvcHostRenderResult {
            RequestId = requestId,
            CacheKey = cacheKey,
            Audio = result.Audio,
            SampleRate = result.SampleRate,
            RenderInfo = result.RenderInfo,
        };
    }
}

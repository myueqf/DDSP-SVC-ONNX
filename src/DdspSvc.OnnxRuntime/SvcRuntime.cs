using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Pipeline;

namespace DdspSvc.OnnxRuntime;

public sealed class SvcRuntime : IDisposable {
    private readonly SvcInferencePipeline pipeline;
    private readonly SvcRenderer renderer;

    private SvcRuntime(SvcInferencePipeline pipeline) {
        this.pipeline = pipeline;
        renderer = new SvcRenderer(pipeline);
    }

    public SvcRuntimeOptions Options => pipeline.Options;
    public SvcModelPaths ModelPaths => pipeline.ModelPaths;
    public VocoderConfig VocoderConfig => pipeline.VocoderConfig;
    public SvcRuntimeCapabilities Capabilities => pipeline.Capabilities;

    public static SvcRuntime Create(SvcRuntimeOptions options, SvcPipelineFactory? factory = null) {
        ArgumentNullException.ThrowIfNull(options);
        factory ??= new SvcPipelineFactory();
        return new SvcRuntime(factory.Create(options));
    }

    public static SvcRuntime Create(SvcInferencePipeline pipeline) {
        ArgumentNullException.ThrowIfNull(pipeline);
        return new SvcRuntime(pipeline);
    }

    public void Warmup() => pipeline.Load();

    public SvcPipelineStatus GetStatus() => pipeline.GetStatus();

    public void ValidateRequest(SvcInferenceRequest request) => renderer.Validate(request);

    public SvcRenderInfo AnalyzeRequest(SvcInferenceRequest request) => renderer.Analyze(request);

    public SvcPreparedConditioning PrepareConditioning(SvcInferenceRequest request) => renderer.PrepareConditioning(request);

    public SvcInferenceResult RenderPrepared(SvcPreparedConditioning conditioning) => renderer.Render(conditioning);

    public SvcInferenceResult Render(SvcInferenceRequest request) => renderer.Render(request);

    public void Dispose() => pipeline.Dispose();
}

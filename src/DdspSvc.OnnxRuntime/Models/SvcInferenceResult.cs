namespace DdspSvc.OnnxRuntime.Models;

public sealed class SvcInferenceResult {
    public required float[] Audio { get; init; }
    public required int SampleRate { get; init; }
    public required SvcRenderInfo RenderInfo { get; init; }
}

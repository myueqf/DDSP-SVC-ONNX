namespace DdspSvc.OnnxRuntime.Models;

public sealed class SvcPreparedConditioning {
    public required int InputSampleRate { get; init; }
    public required int OutputSampleRate { get; init; }
    public required int HopSize { get; init; }
    public int? Seed { get; init; }
    public required int ReflowSteps { get; init; }
    public required int KeyShiftSemitones { get; init; }
    public required float SilenceThresholdDb { get; init; }
    public required int TotalFrames { get; init; }
    public required float[] VolumeMask { get; init; }
    public required float[]? SpeakerMix { get; init; }
    public required IReadOnlyList<SvcSegmentConditioning> Segments { get; init; }
    public required SvcRenderInfo RenderInfo { get; init; }
}

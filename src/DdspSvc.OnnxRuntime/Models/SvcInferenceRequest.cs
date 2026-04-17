namespace DdspSvc.OnnxRuntime.Models;

public sealed class SvcInferenceRequest {
    public required float[] Audio { get; init; }
    public required int SampleRate { get; init; }
    public int? Seed { get; init; }
    public int? ReflowSteps { get; init; }
    public float? SilenceThresholdDb { get; init; }
    public int KeyShiftSemitones { get; init; }
    public int FormantShiftSemitones { get; init; }
    public int VocalRegisterShiftSemitones { get; init; }
    public float[]? SpeakerMix { get; init; }
}

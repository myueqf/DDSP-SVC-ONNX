using DdspSvc.OnnxRuntime.Models;

namespace DdspSvc.OnnxRuntime.Hosting;

public sealed class SvcHostRenderRequest {
    public string? RequestId { get; init; }
    public string? CacheKey { get; init; }
    public required float[] Audio { get; init; }
    public required int SampleRate { get; init; }
    public int? Seed { get; init; }
    public int? ReflowSteps { get; init; }
    public float? SilenceThresholdDb { get; init; }
    public int KeyShiftSemitones { get; init; }
    public int FormantShiftSemitones { get; init; }
    public int VocalRegisterShiftSemitones { get; init; }
    public float[]? SpeakerMix { get; init; }

    public SvcInferenceRequest ToInferenceRequest() {
        return new SvcInferenceRequest {
            Audio = Audio,
            SampleRate = SampleRate,
            Seed = Seed,
            ReflowSteps = ReflowSteps,
            SilenceThresholdDb = SilenceThresholdDb,
            KeyShiftSemitones = KeyShiftSemitones,
            FormantShiftSemitones = FormantShiftSemitones,
            VocalRegisterShiftSemitones = VocalRegisterShiftSemitones,
            SpeakerMix = SpeakerMix,
        };
    }
}

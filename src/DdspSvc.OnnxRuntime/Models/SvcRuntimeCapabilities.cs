namespace DdspSvc.OnnxRuntime.Models;

public sealed record SvcRuntimeCapabilities(
    int SpeakerCount,
    bool SupportsSpeakerMix,
    bool SupportsKeyShift,
    bool SupportsFormantShift,
    bool SupportsVocalRegisterShift,
    bool SupportsNonZeroTStart,
    IReadOnlyList<string> SupportedReflowMethods) {

    public static SvcRuntimeCapabilities FromOptions(SvcRuntimeOptions options) {
        ArgumentNullException.ThrowIfNull(options);
        return new SvcRuntimeCapabilities(
            SpeakerCount: Math.Max(1, options.SpeakerCount),
            SupportsSpeakerMix: options.SpeakerCount > 1,
            SupportsKeyShift: true,
            SupportsFormantShift: false,
            SupportsVocalRegisterShift: false,
            SupportsNonZeroTStart: true,
            SupportedReflowMethods: ["euler", "rk4"]);
    }
}

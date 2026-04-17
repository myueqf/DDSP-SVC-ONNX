namespace DdspSvc.OnnxRuntime.Models;

public sealed record SvcRenderInfo(
    int InputSampleRate,
    int OutputSampleRate,
    int InputSamples,
    int OutputSamples,
    int TotalFrames,
    int SegmentCount,
    int ReflowSteps,
    int KeyShiftSemitones,
    float SilenceThresholdDb,
    bool UsedSpeakerMix);

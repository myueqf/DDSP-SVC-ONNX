namespace DdspSvc.OnnxRuntime.Models;

public sealed class RmvpeResult {
    public double TimeStepSeconds { get; init; } = 0.01;
    public float[] F0Hz { get; init; } = [];
    public bool[] Unvoiced { get; init; } = [];
    public float[] MidiPitch { get; init; } = [];
}

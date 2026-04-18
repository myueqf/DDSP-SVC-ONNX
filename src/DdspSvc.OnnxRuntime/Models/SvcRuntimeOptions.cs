namespace DdspSvc.OnnxRuntime.Models;

public sealed class SvcRuntimeOptions {
    public string ModelRoot { get; init; } = string.Empty;
    public string? DependenciesRoot { get; init; }
    public SvcExecutionProvider ExecutionProvider { get; init; } = SvcExecutionProvider.Cpu;
    public int ExecutionDeviceId { get; init; }
    public string? DDspEncoderPath { get; init; }
    public string? ReflowVelocityPath { get; init; }
    public string? VocoderPath { get; init; }
    public string? RmvpePath { get; init; }
    public string? ContentEncoderPath { get; init; }
    public ContentEncoderKind ContentEncoder { get; init; } = ContentEncoderKind.ContentVec768L12Tta2x;
    public int SamplingRate { get; init; } = 44100;
    public int HopSize { get; init; } = 512;
    public int WinSize { get; init; } = 2048;
    public int MelBins { get; init; } = 128;
    public int ContentEncoderSampleRate { get; init; } = 16000;
    public int ContentEncoderHopSize { get; init; } = 160;
    public float F0MinHz { get; init; } = 65f;
    public float F0MaxHz { get; init; } = 800f;
    public int RmvpeSampleRate { get; init; } = 16000;
    public int RmvpeHopLength { get; init; } = 160;
    public int ReflowSteps { get; init; } = 50;
    public string ReflowMethod { get; init; } = "euler";
    public float ReflowTStart { get; init; }
    public float SilenceThresholdDb { get; init; } = -60f;
    public int SpeakerCount { get; init; } = 1;
    public bool UsePitchAugmentation { get; init; }
}

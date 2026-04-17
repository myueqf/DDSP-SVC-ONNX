namespace DdspSvc.OnnxRuntime.Models;

public sealed class SvcModelConfig {
    public SvcDataConfig Data { get; init; } = new();
    public SvcInferConfig Infer { get; init; } = new();
    public SvcModelSection Model { get; init; } = new();
}

public sealed class SvcDataConfig {
    public int SamplingRate { get; init; } = 44100;
    public int BlockSize { get; init; } = 512;
    public int VolumeSmoothSize { get; init; } = 2048;
    public string Encoder { get; init; } = "contentvec768l12tta2x";
    public int EncoderSampleRate { get; init; } = 16000;
    public int EncoderHopSize { get; init; } = 160;
    public float F0Min { get; init; } = 65f;
    public float F0Max { get; init; } = 800f;
}

public sealed class SvcInferConfig {
    public int InferStep { get; init; } = 50;
    public string Method { get; init; } = "euler";
}

public sealed class SvcModelSection {
    public float? TStart { get; init; }
}

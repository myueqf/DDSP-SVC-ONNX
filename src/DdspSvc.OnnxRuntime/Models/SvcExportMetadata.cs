using System.Text.Json.Serialization;

namespace DdspSvc.OnnxRuntime.Models;

public sealed class SvcExportMetadata {
    [JsonPropertyName("model_path")]
    public string? ModelPath { get; init; }

    [JsonPropertyName("sampling_rate")]
    public int SamplingRate { get; init; }

    [JsonPropertyName("block_size")]
    public int BlockSize { get; init; }

    [JsonPropertyName("win_length")]
    public int WinLength { get; init; }

    [JsonPropertyName("encoder")]
    public string? Encoder { get; init; }

    [JsonPropertyName("encoder_out_channels")]
    public int EncoderOutChannels { get; init; }

    [JsonPropertyName("n_spk")]
    public int? SpeakerCount { get; init; }

    [JsonPropertyName("use_pitch_aug")]
    public bool UsePitchAugmentation { get; init; }

    [JsonPropertyName("onnx")]
    public SvcExportOnnxFiles Onnx { get; init; } = new();
}

public sealed class SvcExportOnnxFiles {
    [JsonPropertyName("encoder")]
    public string Encoder { get; init; } = "encoder.onnx";

    [JsonPropertyName("velocity")]
    public string Velocity { get; init; } = "velocity.onnx";
}

using YamlDotNet.Serialization;

namespace DdspSvc.OnnxRuntime.Models;

public sealed class VocoderConfig {
    [YamlMember(Alias = "name")]
    public string Name { get; init; } = "vocoder";

    [YamlMember(Alias = "model")]
    public string Model { get; init; } = "model.onnx";

    [YamlMember(Alias = "sample_rate")]
    public int SampleRate { get; init; } = 44100;

    [YamlMember(Alias = "hop_size")]
    public int HopSize { get; init; } = 512;

    [YamlMember(Alias = "win_size")]
    public int WinSize { get; init; } = 2048;

    [YamlMember(Alias = "fft_size")]
    public int FftSize { get; init; } = 2048;

    [YamlMember(Alias = "num_mel_bins")]
    public int NumMelBins { get; init; } = 128;

    [YamlMember(Alias = "mel_fmin")]
    public double MelFmin { get; init; } = 40;

    [YamlMember(Alias = "mel_fmax")]
    public double MelFmax { get; init; } = 16000;

    [YamlMember(Alias = "mel_base")]
    public string MelBase { get; init; } = "10";

    [YamlMember(Alias = "mel_scale")]
    public string MelScale { get; init; } = "slaney";

    [YamlMember(Alias = "pitch_controllable")]
    public bool PitchControllable { get; init; }
}

using Microsoft.ML.OnnxRuntime;
using DdspSvc.OnnxRuntime.Audio;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Onnx;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class SvcInferencePipeline : IDisposable {
    private readonly OnnxSessionFactory sessionFactory;
    private readonly VolumeExtractor volumeExtractor;
    private readonly VocoderConfig vocoderConfig;
    private ContentVecEncoder? contentEncoder;
    private DdspEncoder? ddspEncoder;
    private ReflowSampler? reflowSampler;
    private RmvpePitchExtractor? rmvpeExtractor;
    private NsfHifiganVocoder? vocoder;

    public SvcRuntimeOptions Options { get; }
    public SvcModelPaths ModelPaths { get; }
    public VocoderConfig VocoderConfig => vocoderConfig;
    public SvcRuntimeCapabilities Capabilities { get; }

    public SvcInferencePipeline(
        SvcRuntimeOptions options,
        SvcModelPaths modelPaths,
        VocoderConfig vocoderConfig,
        OnnxSessionFactory? sessionFactory = null) {
        Options = options;
        ModelPaths = modelPaths;
        this.vocoderConfig = vocoderConfig;
        this.sessionFactory = sessionFactory ?? new OnnxSessionFactory();
        volumeExtractor = new VolumeExtractor(options.HopSize, options.WinSize);
        Capabilities = SvcRuntimeCapabilities.FromOptions(options);
    }

    public void Load() {
        contentEncoder ??= new ContentVecEncoder(
            ModelPaths.ContentEncoderPath,
            Options.ContentEncoder,
            Options.ContentEncoderSampleRate,
            Options.ContentEncoderHopSize,
            sessionFactory);
        rmvpeExtractor ??= new RmvpePitchExtractor(ModelPaths.RmvpePath, sessionFactory);
        ddspEncoder ??= new DdspEncoder(ModelPaths.DDspEncoderPath, Options.HopSize, sessionFactory);
        reflowSampler ??= new ReflowSampler(ModelPaths.ReflowVelocityPath, sessionFactory);
        vocoder ??= new NsfHifiganVocoder(ModelPaths.VocoderPath, vocoderConfig, sessionFactory);
    }

    public float[] ExtractVolume(ReadOnlySpan<float> audio, int sampleRate) {
        var normalized = sampleRate == VocoderConfig.SampleRate
            ? audio.ToArray()
            : LinearResampler.Resample(audio, sampleRate, VocoderConfig.SampleRate);
        return volumeExtractor.Extract(normalized);
    }

    public ContentUnitsResult ExtractUnits(ReadOnlySpan<float> audio, int sampleRate) {
        Load();
        return contentEncoder!.Encode(audio, sampleRate, Options.HopSize);
    }

    public RmvpeResult ExtractPitch(ReadOnlySpan<float> audio, int sampleRate) {
        Load();
        return rmvpeExtractor!.Infer(audio, sampleRate, Options.HopSize, Options.F0MinHz);
    }

    public DdspEncoderResult EncodeDdsp(
        ContentUnitsResult units,
        float[] f0,
        float[] volume,
        int? seed = null,
        float[]? speakerMix = null) {
        Load();
        return ddspEncoder!.Infer(units, f0, volume, seed, speakerMix);
    }

    public float[] SampleReflow(DdspEncoderResult ddsp, int steps = 20, int? seed = null) {
        Load();
        return reflowSampler!.Sample(ddsp, steps, Options.ReflowMethod, Options.ReflowTStart, seed);
    }

    public VocoderResult VocoderInfer(float[] mel, int melFrames, int melBins, float[] f0) {
        Load();
        return vocoder!.Infer(mel, melFrames, melBins, f0);
    }

    public SvcPipelineStatus GetStatus() {
        return new SvcPipelineStatus(
            ContentEncoderLoaded: contentEncoder is not null,
            RmvpeLoaded: rmvpeExtractor is not null,
            DDspEncoderLoaded: ddspEncoder is not null,
            VelocityLoaded: reflowSampler is not null,
            VocoderLoaded: vocoder is not null);
    }

    public void Dispose() {
        contentEncoder?.Dispose();
        rmvpeExtractor?.Dispose();
        ddspEncoder?.Dispose();
        reflowSampler?.Dispose();
        vocoder?.Dispose();
    }
}

public sealed record SvcPipelineStatus(
    bool ContentEncoderLoaded,
    bool RmvpeLoaded,
    bool DDspEncoderLoaded,
    bool VelocityLoaded,
    bool VocoderLoaded);

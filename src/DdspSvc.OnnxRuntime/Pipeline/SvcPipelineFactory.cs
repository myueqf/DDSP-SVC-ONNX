using DdspSvc.OnnxRuntime.Models;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class SvcPipelineFactory {
    private readonly SvcAssetResolver assetResolver = new();

    public SvcInferencePipeline Create(SvcRuntimeOptions options) {
        var resolvedOptions = ResolveOptions(options);
        var modelPaths = assetResolver.Resolve(resolvedOptions);
        var vocoderConfig = assetResolver.LoadVocoderConfig(Path.Combine(Path.GetDirectoryName(modelPaths.VocoderPath)!, "vocoder.yaml"));
        return new SvcInferencePipeline(resolvedOptions, modelPaths, vocoderConfig);
    }

    public SvcRuntimeOptions ResolveOptions(SvcRuntimeOptions options) {
        var modelConfig = assetResolver.TryLoadModelConfig(options.ModelRoot);
        var exportMetadata = assetResolver.TryLoadExportMetadata(options.ModelRoot);

        var encoder = exportMetadata?.Encoder ?? modelConfig?.Data.Encoder;
        var contentEncoder = encoder switch {
            "contentvec768l12" => ContentEncoderKind.ContentVec768L12,
            "contentvec768l12tta2x" => ContentEncoderKind.ContentVec768L12Tta2x,
            "hubertsoft" => ContentEncoderKind.HubertSoft,
            _ => options.ContentEncoder,
        };

        return new SvcRuntimeOptions {
            ModelRoot = options.ModelRoot,
            DependenciesRoot = options.DependenciesRoot,
            ExecutionProvider = options.ExecutionProvider,
            ExecutionDeviceId = options.ExecutionDeviceId,
            DDspEncoderPath = options.DDspEncoderPath,
            ReflowVelocityPath = options.ReflowVelocityPath,
            VocoderPath = options.VocoderPath,
            RmvpePath = options.RmvpePath,
            ContentEncoderPath = options.ContentEncoderPath,
            ContentEncoder = contentEncoder,
            SamplingRate = modelConfig?.Data.SamplingRate ?? exportMetadata?.SamplingRate ?? options.SamplingRate,
            HopSize = modelConfig?.Data.BlockSize ?? exportMetadata?.BlockSize ?? options.HopSize,
            WinSize = modelConfig?.Data.VolumeSmoothSize ?? options.WinSize,
            MelBins = options.MelBins,
            ContentEncoderSampleRate = modelConfig?.Data.EncoderSampleRate ?? options.ContentEncoderSampleRate,
            ContentEncoderHopSize = modelConfig?.Data.EncoderHopSize ?? options.ContentEncoderHopSize,
            F0MinHz = modelConfig?.Data.F0Min ?? options.F0MinHz,
            F0MaxHz = modelConfig?.Data.F0Max ?? options.F0MaxHz,
            RmvpeSampleRate = options.RmvpeSampleRate,
            RmvpeHopLength = options.RmvpeHopLength,
            ReflowSteps = modelConfig?.Infer.InferStep ?? options.ReflowSteps,
            ReflowMethod = modelConfig?.Infer.Method ?? options.ReflowMethod,
            ReflowTStart = modelConfig?.Model.TStart ?? options.ReflowTStart,
            SilenceThresholdDb = options.SilenceThresholdDb,
            SpeakerCount = exportMetadata?.SpeakerCount ?? options.SpeakerCount,
            UsePitchAugmentation = exportMetadata?.UsePitchAugmentation ?? options.UsePitchAugmentation,
        };
    }
}

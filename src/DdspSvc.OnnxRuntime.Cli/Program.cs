using DdspSvc.OnnxRuntime.Audio;
using DdspSvc.OnnxRuntime;
using DdspSvc.OnnxRuntime.Models;
using DdspSvc.OnnxRuntime.Pipeline;
using System.Globalization;
using System.Diagnostics;

if (args.Length == 0) {
    PrintUsage();
    return 1;
}

var command = args[0].ToLowerInvariant();
if (command != "inspect" && command != "smoke" && command != "render") {
    PrintUsage();
    return 1;
}

try {
    var factory = new SvcPipelineFactory();

    if (command == "inspect") {
        var inspectArgs = ParseInspectArgs(args.Skip(1).ToArray());
        var options = factory.ResolveOptions(new SvcRuntimeOptions {
            ModelRoot = inspectArgs.ModelRoot,
            DependenciesRoot = inspectArgs.DependenciesRoot,
            DDspEncoderPath = inspectArgs.DDspEncoderPath,
            ReflowVelocityPath = inspectArgs.ReflowVelocityPath,
            VocoderPath = inspectArgs.VocoderPath,
            RmvpePath = inspectArgs.RmvpePath,
            ContentEncoderPath = inspectArgs.ContentEncoderPath,
        });
        var resolver = new SvcAssetResolver();
        var dependenciesRoot = resolver.ResolveDependenciesRootPath(options);
        var paths = resolver.Resolve(options);
        var vocoderConfig = resolver.LoadVocoderConfig(Path.Combine(Path.GetDirectoryName(paths.VocoderPath)!, "vocoder.yaml"));

        Console.WriteLine("Resolved assets:");
        Console.WriteLine($"  Dependencies : {dependenciesRoot}");
        Console.WriteLine($"  DDSP encoder : {paths.DDspEncoderPath}");
        Console.WriteLine($"  Reflow       : {paths.ReflowVelocityPath}");
        Console.WriteLine($"  Vocoder      : {paths.VocoderPath}");
        Console.WriteLine($"  RMVPE        : {paths.RmvpePath}");
        Console.WriteLine($"  Content      : {paths.ContentEncoderPath}");
        Console.WriteLine();
        Console.WriteLine("Resolved options:");
        Console.WriteLine($"  Encoder      : {options.ContentEncoder}");
        Console.WriteLine($"  Sample rate  : {options.SamplingRate}");
        Console.WriteLine($"  Hop size     : {options.HopSize}");
        Console.WriteLine($"  Win size     : {options.WinSize}");
        Console.WriteLine($"  F0 min/max   : {options.F0MinHz}/{options.F0MaxHz}");
        Console.WriteLine($"  Speakers     : {options.SpeakerCount}");
        Console.WriteLine($"  Pitch aug    : {options.UsePitchAugmentation}");
        Console.WriteLine($"  Reflow       : {options.ReflowMethod} {options.ReflowSteps} steps t={options.ReflowTStart}");
        Console.WriteLine();
        Console.WriteLine("Vocoder config:");
        Console.WriteLine($"  Sample rate  : {vocoderConfig.SampleRate}");
        Console.WriteLine($"  Hop size     : {vocoderConfig.HopSize}");
        Console.WriteLine($"  Mel bins     : {vocoderConfig.NumMelBins}");
        Console.WriteLine($"  Pitch ctrl   : {vocoderConfig.PitchControllable}");
    } else {
        var renderArgs = ParseRenderArgs(args.Skip(1).ToArray());
        var options = factory.ResolveOptions(new SvcRuntimeOptions {
            ModelRoot = renderArgs.ModelRoot,
            DependenciesRoot = renderArgs.DependenciesRoot,
            DDspEncoderPath = renderArgs.DDspEncoderPath,
            ReflowVelocityPath = renderArgs.ReflowVelocityPath,
            VocoderPath = renderArgs.VocoderPath,
            RmvpePath = renderArgs.RmvpePath,
            ContentEncoderPath = renderArgs.ContentEncoderPath,
        });
        using var runtime = SvcRuntime.Create(options, factory);
        RenderLongAudio(runtime, renderArgs.InputWav, renderArgs.OutputWav, renderArgs.Options);
        Console.WriteLine($"Wrote {renderArgs.OutputWav}");
    }
    return 0;
} catch (Exception ex) {
    Console.Error.WriteLine(ex.Message);
    return 2;
}

static void PrintUsage() {
    var cwd = Directory.GetCurrentDirectory();
    var defaultModelRoot = GetDefaultModelRoot();
    var defaultDependenciesRoot = GetDefaultDependenciesRoot();

    Console.WriteLine("Usage:");
    Console.WriteLine("  ddspsvc-onnx inspect [--model-root PATH] [--dependencies-root PATH]");
    Console.WriteLine("  ddspsvc-onnx inspect <model-root> [dependencies-root]");
    Console.WriteLine("  ddspsvc-onnx render [--model-root PATH] [--dependencies-root PATH] <input-wav> <output-wav> [options]");
    Console.WriteLine("  ddspsvc-onnx render <model-root> <dependencies-root> <input-wav> <output-wav> [options]");
    Console.WriteLine("  ddspsvc-onnx smoke  [--model-root PATH] [--dependencies-root PATH] <input-wav> <output-wav> [options]");
    Console.WriteLine();
    Console.WriteLine("Default Layout:");
    Console.WriteLine($"  Working dir         : {cwd}");
    Console.WriteLine($"  Model root          : {defaultModelRoot}");
    Console.WriteLine($"  Dependencies root   : {defaultDependenciesRoot}");
    Console.WriteLine("  Expected structure  :");
    Console.WriteLine("    ./Model/onnx/encoder.onnx");
    Console.WriteLine("    ./Model/onnx/velocity.onnx");
    Console.WriteLine("    ./Model/onnx/svc.json");
    Console.WriteLine("    ./Dependencies/rmvpe/rmvpe.onnx");
    Console.WriteLine("    ./Dependencies/contentvec/contentvec.onnx");
    Console.WriteLine("    ./Dependencies/pc_nsf_hifigan_*/vocoder.yaml");
    Console.WriteLine("    ./Dependencies/pc_nsf_hifigan_*/<model>.onnx");
    Console.WriteLine();
    Console.WriteLine("Render Defaults:");
    Console.WriteLine("  --chunk-seconds      48");
    Console.WriteLine("  --overlap-seconds    1");
    Console.WriteLine("  --seam-crossfade-ms  50");
    Console.WriteLine("  --slice-threshold-db -40");
    Console.WriteLine("  --threshold-db       -60");
    Console.WriteLine("  --seed               0");
    Console.WriteLine("  --key                0");
    Console.WriteLine("  --reflow-steps       model default");
    Console.WriteLine();
    Console.WriteLine("Notes:");
    Console.WriteLine("  Long-audio rendering is silence-slice first.");
    Console.WriteLine("  --chunk-seconds only applies to oversized voiced slices after silence slicing.");
    Console.WriteLine("  --slice-threshold-db controls silence slicing.");
    Console.WriteLine("  --threshold-db controls the final loudness mask inside inference.");
    Console.WriteLine("  Explicit asset overrides are available: --encoder-path, --velocity-path, --contentvec-path, --rmvpe-path, --vocoder-path.");
}

static InspectCliArgs ParseInspectArgs(string[] args) {
    string? modelRoot = null;
    string? dependenciesRoot = null;
    string? encoderPath = null;
    string? velocityPath = null;
    string? vocoderPath = null;
    string? rmvpePath = null;
    string? contentvecPath = null;
    var positional = new List<string>();

    for (var i = 0; i < args.Length; i++) {
        switch (args[i]) {
            case "--model-root":
                modelRoot = ParsePath(args, ref i, "--model-root");
                break;
            case "--dependencies-root":
                dependenciesRoot = ParsePath(args, ref i, "--dependencies-root");
                break;
            case "--encoder-path":
                encoderPath = ParsePath(args, ref i, "--encoder-path");
                break;
            case "--velocity-path":
                velocityPath = ParsePath(args, ref i, "--velocity-path");
                break;
            case "--vocoder-path":
                vocoderPath = ParsePath(args, ref i, "--vocoder-path");
                break;
            case "--rmvpe-path":
                rmvpePath = ParsePath(args, ref i, "--rmvpe-path");
                break;
            case "--contentvec-path":
                contentvecPath = ParsePath(args, ref i, "--contentvec-path");
                break;
            default:
                positional.Add(args[i]);
                break;
        }
    }

    if (positional.Count > 2) {
        throw new ArgumentException("inspect accepts at most <model-root> [dependencies-root] after options.");
    }

    if (positional.Count >= 1) {
        modelRoot ??= Path.GetFullPath(positional[0]);
    }
    if (positional.Count >= 2) {
        dependenciesRoot ??= Path.GetFullPath(positional[1]);
    }

    return new InspectCliArgs(
        ModelRoot: modelRoot ?? GetDefaultModelRoot(),
        DependenciesRoot: dependenciesRoot ?? GetDefaultDependenciesRoot(),
        DDspEncoderPath: encoderPath,
        ReflowVelocityPath: velocityPath,
        VocoderPath: vocoderPath,
        RmvpePath: rmvpePath,
        ContentEncoderPath: contentvecPath);
}

static RenderCliArgs ParseRenderArgs(string[] args) {
    var options = new RenderCliOptions();
    string? modelRoot = null;
    string? dependenciesRoot = null;
    string? encoderPath = null;
    string? velocityPath = null;
    string? vocoderPath = null;
    string? rmvpePath = null;
    string? contentvecPath = null;
    var positional = new List<string>();

    for (var i = 0; i < args.Length; i++) {
        switch (args[i]) {
            case "--model-root":
                modelRoot = ParsePath(args, ref i, "--model-root");
                break;
            case "--dependencies-root":
                dependenciesRoot = ParsePath(args, ref i, "--dependencies-root");
                break;
            case "--encoder-path":
                encoderPath = ParsePath(args, ref i, "--encoder-path");
                break;
            case "--velocity-path":
                velocityPath = ParsePath(args, ref i, "--velocity-path");
                break;
            case "--vocoder-path":
                vocoderPath = ParsePath(args, ref i, "--vocoder-path");
                break;
            case "--rmvpe-path":
                rmvpePath = ParsePath(args, ref i, "--rmvpe-path");
                break;
            case "--contentvec-path":
                contentvecPath = ParsePath(args, ref i, "--contentvec-path");
                break;
            case "--chunk-seconds":
                options = options with { ChunkSeconds = ParsePositiveFloat(args, ref i, "--chunk-seconds") };
                break;
            case "--overlap-seconds":
                options = options with { OverlapSeconds = ParseNonNegativeFloat(args, ref i, "--overlap-seconds") };
                break;
            case "--seam-crossfade-ms":
                options = options with { SeamCrossfadeSeconds = ParseNonNegativeFloat(args, ref i, "--seam-crossfade-ms") / 1000f };
                break;
            case "--slice-threshold-db":
                options = options with { SliceThresholdDb = ParseFloat(args, ref i, "--slice-threshold-db") };
                break;
            case "--seed":
                options = options with { Seed = ParseInt(args, ref i, "--seed") };
                break;
            case "--key":
                options = options with { KeyShiftSemitones = ParseInt(args, ref i, "--key") };
                break;
            case "--threshold-db":
                options = options with { SilenceThresholdDb = ParseFloat(args, ref i, "--threshold-db") };
                break;
            case "--reflow-steps":
                options = options with { ReflowSteps = ParsePositiveInt(args, ref i, "--reflow-steps") };
                break;
            case "--speaker-mix":
                options = options with { SpeakerMix = ParseSpeakerMix(args, ref i, "--speaker-mix") };
                break;
            default:
                positional.Add(args[i]);
                break;
        }
    }

    if (options.OverlapSeconds * 2 >= options.ChunkSeconds) {
        throw new ArgumentException("--overlap-seconds must be less than half of --chunk-seconds.");
    }

    string inputWav;
    string outputWav;
    if (positional.Count == 2) {
        inputWav = Path.GetFullPath(positional[0]);
        outputWav = Path.GetFullPath(positional[1]);
    } else if (positional.Count == 4) {
        modelRoot ??= Path.GetFullPath(positional[0]);
        dependenciesRoot ??= Path.GetFullPath(positional[1]);
        inputWav = Path.GetFullPath(positional[2]);
        outputWav = Path.GetFullPath(positional[3]);
    } else {
        throw new ArgumentException(
            "render/smoke expects either <input-wav> <output-wav> with default roots, or <model-root> <dependencies-root> <input-wav> <output-wav>.");
    }

    return new RenderCliArgs(
        ModelRoot: modelRoot ?? GetDefaultModelRoot(),
        DependenciesRoot: dependenciesRoot ?? GetDefaultDependenciesRoot(),
        DDspEncoderPath: encoderPath,
        ReflowVelocityPath: velocityPath,
        VocoderPath: vocoderPath,
        RmvpePath: rmvpePath,
        ContentEncoderPath: contentvecPath,
        InputWav: inputWav,
        OutputWav: outputWav,
        Options: options);
}

static void RenderLongAudio(SvcRuntime runtime, string inputWav, string outputWav, RenderCliOptions options) {
    var wav = WaveFile.ReadMono16(inputWav);
    var totalOutputSamples = GetResampledSampleCount(wav.Samples.Length, wav.SampleRate, runtime.VocoderConfig.SampleRate);
    var slicer = new AudioSlicer(
        wav.SampleRate,
        thresholdDb: options.SliceThresholdDb);
    var slices = slicer.Slice(wav.Samples);
    var voicedSlices = slices
        .Where(slice => !slice.IsSilence && slice.EndSample > slice.StartSample)
        .ToArray();

    var stitched = Array.Empty<float>();
    var currentLength = 0;
    var totalVoiced = voicedSlices.Length;

    for (var i = 0; i < totalVoiced; i++) {
        var slice = voicedSlices[i];
        var segment = Slice(wav.Samples, slice.StartSample, slice.EndSample - slice.StartSample);
        var segmentSeconds = segment.Length / (double)wav.SampleRate;
        var remaining = totalVoiced - i - 1;
        var watch = Stopwatch.StartNew();
        Console.WriteLine($"{BuildProgressBar(i, totalVoiced)} slice {i + 1}/{totalVoiced}  remaining {remaining}  voiced {segmentSeconds:F2}s");
        var output = segment.Length <= Math.Round(options.ChunkSeconds * wav.SampleRate)
            ? RenderClip(runtime, segment, wav.SampleRate, options)
            : RenderOversizedVoicedSlice(runtime, segment, wav.SampleRate, options);
        watch.Stop();

        var startOutputSample = GetResampledSampleCount(slice.StartSample, wav.SampleRate, runtime.VocoderConfig.SampleRate);
        stitched = AppendAt(stitched, output, startOutputSample, ref currentLength);
        Console.WriteLine($"{BuildProgressBar(i + 1, totalVoiced)} done {watch.Elapsed.TotalSeconds:F2}s");
    }

    stitched = MatchLength(stitched, totalOutputSamples);
    WaveFile.WriteMono16(outputWav, stitched, runtime.VocoderConfig.SampleRate);
}

static float[] RenderClip(SvcRuntime runtime, float[] audio, int sampleRate, RenderCliOptions options) {
    var result = runtime.Render(new SvcInferenceRequest {
        Audio = audio,
        SampleRate = sampleRate,
        Seed = options.Seed,
        ReflowSteps = options.ReflowSteps,
        SilenceThresholdDb = options.SilenceThresholdDb,
        KeyShiftSemitones = options.KeyShiftSemitones,
        SpeakerMix = options.SpeakerMix,
    });
    return result.Audio;
}

static float[] RenderOversizedVoicedSlice(SvcRuntime runtime, float[] audio, int sampleRate, RenderCliOptions options) {
    var chunkInputSamples = Math.Max(1, (int)Math.Round(options.ChunkSeconds * sampleRate));
    var overlapInputSamples = Math.Max(0, (int)Math.Round(options.OverlapSeconds * sampleRate));
    var outputSampleRate = runtime.VocoderConfig.SampleRate;
    var seamCrossfadeSamples = Math.Max(1, (int)Math.Round(options.SeamCrossfadeSeconds * outputSampleRate));
    var cutSearchRadiusSamples = Math.Max(1, chunkInputSamples / 2);

    var stitched = Array.Empty<float>();
    var currentLength = 0;
    float[] leftContext = [];
    var offset = 0;

    while (offset < audio.Length) {
        var remainingInputSamples = audio.Length - offset;
        var mainLength = Math.Min(chunkInputSamples, remainingInputSamples);
        if (remainingInputSamples > chunkInputSamples) {
            var targetCut = offset + chunkInputSamples;
            var cut = FindWeakEnergyCut(
                audio,
                sampleRate,
                targetCut,
                offset + Math.Max(1, chunkInputSamples / 2),
                Math.Min(audio.Length, targetCut + cutSearchRadiusSamples));
            mainLength = Math.Max(1, cut - offset);
        }
        var mainChunk = Slice(audio, offset, mainLength);
        var rightLength = Math.Min(overlapInputSamples, Math.Max(0, audio.Length - (offset + mainLength)));
        var rightContext = rightLength > 0
            ? Slice(audio, offset + mainLength, rightLength)
            : [];
        var input = ConcatThree(leftContext, mainChunk, rightContext);
        var rendered = RenderClip(runtime, input, sampleRate, options);

        var leftTrimOutputSamples = GetResampledSampleCount(leftContext.Length, sampleRate, outputSampleRate);
        var mainOutputSamples = GetResampledSampleCount(mainChunk.Length, sampleRate, outputSampleRate);
        var centered = SliceWithEdgePadding(rendered, leftTrimOutputSamples, mainOutputSamples);
        stitched = AppendWithBoundaryFade(stitched, centered, ref currentLength, seamCrossfadeSamples);

        leftContext = overlapInputSamples > 0
            ? Tail(mainChunk, overlapInputSamples)
            : [];
        offset += mainLength;
    }

    return stitched;
}

static int FindWeakEnergyCut(float[] audio, int sampleRate, int targetSample, int minSample, int maxSample) {
    if (maxSample <= minSample) {
        return Math.Clamp(targetSample, minSample, maxSample);
    }

    var hop = Math.Max(1, (int)Math.Round(sampleRate * 0.02));
    var win = Math.Max(hop, Math.Min((int)Math.Round(sampleRate * 0.3), hop * 4));
    var bestCut = Math.Clamp(targetSample, minSample, maxSample);
    var bestEnergy = double.PositiveInfinity;

    var frameStart = Math.Max(0, minSample / hop);
    var frameEnd = Math.Max(frameStart, maxSample / hop);
    for (var frame = frameStart; frame <= frameEnd; frame++) {
        var center = frame * hop;
        var energy = ComputeWindowEnergy(audio, center, win);
        var distancePenalty = Math.Abs(center - targetSample) / (double)Math.Max(1, maxSample - minSample);
        var score = energy + distancePenalty * 1e-6;
        if (score < bestEnergy) {
            bestEnergy = score;
            bestCut = center;
        }
    }

    return Math.Clamp(bestCut, minSample, maxSample);
}

static double ComputeWindowEnergy(float[] audio, int center, int windowSize) {
    if (audio.Length == 0) {
        return 0d;
    }

    var half = windowSize / 2;
    var start = Math.Max(0, center - half);
    var end = Math.Min(audio.Length, start + windowSize);
    start = Math.Max(0, end - windowSize);

    double sumSquares = 0d;
    for (var i = start; i < end; i++) {
        sumSquares += audio[i] * audio[i];
    }

    var count = Math.Max(1, end - start);
    return sumSquares / count;
}

static float[] AppendAt(float[] existing, float[] clip, int startSample, ref int currentLength) {
    var silentLength = startSample - currentLength;
    if (silentLength >= 0) {
        var result = new float[existing.Length + silentLength + clip.Length];
        Array.Copy(existing, result, existing.Length);
        Array.Copy(clip, 0, result, existing.Length + silentLength, clip.Length);
        currentLength += silentLength + clip.Length;
        return result;
    }

    var overlapStart = currentLength + silentLength;
    var blended = CrossFade(existing, clip, overlapStart);
    currentLength = overlapStart + clip.Length;
    return blended;
}

static float[] AppendWithBoundaryFade(float[] existing, float[] clip, ref int currentLength, int seamFadeSamples) {
    if (existing.Length == 0) {
        currentLength = clip.Length;
        return (float[])clip.Clone();
    }

    var result = new float[existing.Length + clip.Length];
    Array.Copy(existing, result, existing.Length);

    var blendedClip = (float[])clip.Clone();
    var fade = Math.Min(Math.Min(existing.Length, blendedClip.Length), seamFadeSamples);
    if (fade > 1) {
        for (var i = 0; i < fade; i++) {
            var t = i / (float)(fade - 1);
            var fadeOut = 1f - t;
            var fadeIn = t;
            result[existing.Length - fade + i] *= fadeOut;
            blendedClip[i] *= fadeIn;
        }
    }

    Array.Copy(blendedClip, 0, result, existing.Length, blendedClip.Length);
    currentLength += blendedClip.Length;
    return result;
}

static float[] CrossFade(float[] left, float[] right, int index) {
    if (left.Length == 0) {
        return (float[])right.Clone();
    }
    if (right.Length == 0) {
        return left;
    }

    var safeIndex = Math.Clamp(index, 0, left.Length);
    var fadeLength = left.Length - safeIndex;
    if (fadeLength <= 0) {
        return ConcatTwo(left, right);
    }

    var result = new float[safeIndex + right.Length];
    Array.Copy(left, result, safeIndex);
    var overlap = Math.Min(fadeLength, right.Length);
    for (var i = 0; i < overlap; i++) {
        var t = overlap <= 1 ? 1f : i / (float)(overlap - 1);
        result[safeIndex + i] = (1f - t) * left[safeIndex + i] + t * right[i];
    }
    if (right.Length > overlap) {
        Array.Copy(right, overlap, result, safeIndex + overlap, right.Length - overlap);
    }
    return result;
}

static float[] MatchLength(float[] audio, int targetLength) {
    if (targetLength <= 0) {
        return [];
    }
    if (audio.Length == targetLength) {
        return audio;
    }
    if (audio.Length > targetLength) {
        return audio[..targetLength];
    }

    var result = new float[targetLength];
    Array.Copy(audio, result, audio.Length);
    return result;
}

static float[] ConcatTwo(float[] prefix, float[] suffix) {
    if (prefix.Length == 0) {
        return suffix;
    }
    var result = new float[prefix.Length + suffix.Length];
    Array.Copy(prefix, result, prefix.Length);
    Array.Copy(suffix, 0, result, prefix.Length, suffix.Length);
    return result;
}

static float[] ConcatThree(float[] first, float[] second, float[] third) {
    if (first.Length == 0) {
        return ConcatTwo(second, third);
    }
    if (third.Length == 0) {
        return ConcatTwo(first, second);
    }

    var result = new float[first.Length + second.Length + third.Length];
    Array.Copy(first, result, first.Length);
    Array.Copy(second, 0, result, first.Length, second.Length);
    Array.Copy(third, 0, result, first.Length + second.Length, third.Length);
    return result;
}

static float[] Slice(float[] source, int start, int length) {
    if (length <= 0) {
        return [];
    }
    var result = new float[length];
    Array.Copy(source, start, result, 0, length);
    return result;
}

static float[] SliceWithEdgePadding(float[] source, int start, int length) {
    if (length <= 0) {
        return [];
    }

    var result = new float[length];
    if (source.Length == 0) {
        return result;
    }

    var safeStart = Math.Clamp(start, 0, source.Length - 1);
    var available = Math.Max(0, Math.Min(length, source.Length - safeStart));
    if (available > 0) {
        Array.Copy(source, safeStart, result, 0, available);
    }

    var fillValue = available > 0 ? result[available - 1] : source[^1];
    for (var i = available; i < length; i++) {
        result[i] = fillValue;
    }
    return result;
}

static float[] Tail(float[] source, int length) {
    if (length <= 0 || source.Length == 0) {
        return [];
    }
    var actualLength = Math.Min(length, source.Length);
    return Slice(source, source.Length - actualLength, actualLength);
}

static int GetResampledSampleCount(int sourceSampleCount, int sourceSampleRate, int targetSampleRate) {
    if (sourceSampleCount <= 0) {
        return 0;
    }
    return Math.Max(1, (int)Math.Round(sourceSampleCount * (double)targetSampleRate / sourceSampleRate));
}

static string BuildProgressBar(int completed, int total, int width = 24) {
    if (total <= 0) {
        return "[------------------------]";
    }

    var filled = (int)Math.Round(completed / (double)total * width);
    filled = Math.Clamp(filled, 0, width);
    return $"[{new string('#', filled)}{new string('-', width - filled)}]";
}

static float ParsePositiveFloat(string[] args, ref int index, string option) {
    var value = ParseFloat(args, ref index, option);
    if (value <= 0f) {
        throw new ArgumentException($"{option} must be positive.");
    }
    return value;
}

static string ParsePath(string[] args, ref int index, string option) {
    if (index + 1 >= args.Length) {
        throw new ArgumentException($"Missing path value for {option}.");
    }
    index++;
    return Path.GetFullPath(args[index]);
}

static float ParseNonNegativeFloat(string[] args, ref int index, string option) {
    var value = ParseFloat(args, ref index, option);
    if (value < 0f) {
        throw new ArgumentException($"{option} must be non-negative.");
    }
    return value;
}

static float ParseFloat(string[] args, ref int index, string option) {
    if (index + 1 >= args.Length || !float.TryParse(args[index + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var value)) {
        throw new ArgumentException($"Missing numeric value for {option}.");
    }
    index++;
    return value;
}

static int ParseInt(string[] args, ref int index, string option) {
    if (index + 1 >= args.Length || !int.TryParse(args[index + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)) {
        throw new ArgumentException($"Missing integer value for {option}.");
    }
    index++;
    return value;
}

static int ParsePositiveInt(string[] args, ref int index, string option) {
    var value = ParseInt(args, ref index, option);
    if (value <= 0) {
        throw new ArgumentException($"{option} must be positive.");
    }
    return value;
}

static float[] ParseSpeakerMix(string[] args, ref int index, string option) {
    if (index + 1 >= args.Length) {
        throw new ArgumentException($"Missing value for {option}.");
    }
    index++;
    var parts = args[index].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    if (parts.Length == 0) {
        throw new ArgumentException($"{option} must contain at least one weight.");
    }

    var values = new float[parts.Length];
    for (var i = 0; i < parts.Length; i++) {
        if (!float.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i])) {
            throw new ArgumentException($"Invalid speaker mix weight '{parts[i]}'.");
        }
    }
    return values;
}

static string GetDefaultModelRoot() => Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "Model", "onnx"));

static string GetDefaultDependenciesRoot() => Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "Dependencies"));

internal sealed record InspectCliArgs(
    string ModelRoot,
    string DependenciesRoot,
    string? DDspEncoderPath,
    string? ReflowVelocityPath,
    string? VocoderPath,
    string? RmvpePath,
    string? ContentEncoderPath);

internal sealed record RenderCliArgs(
    string ModelRoot,
    string DependenciesRoot,
    string? DDspEncoderPath,
    string? ReflowVelocityPath,
    string? VocoderPath,
    string? RmvpePath,
    string? ContentEncoderPath,
    string InputWav,
    string OutputWav,
    RenderCliOptions Options);

internal sealed record RenderCliOptions(
    float ChunkSeconds = 48f,
    float OverlapSeconds = 1f,
    float SeamCrossfadeSeconds = 0.05f,
    float SliceThresholdDb = -40f,
    int? Seed = 0,
    int? ReflowSteps = null,
    float? SilenceThresholdDb = -60f,
    int KeyShiftSemitones = 0,
    float[]? SpeakerMix = null);

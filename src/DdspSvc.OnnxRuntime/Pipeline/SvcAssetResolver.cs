using DdspSvc.OnnxRuntime.Models;
using System.Text.Json;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace DdspSvc.OnnxRuntime.Pipeline;

public sealed class SvcAssetResolver {
    private readonly IDeserializer yamlDeserializer = new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .IgnoreUnmatchedProperties()
        .Build();
    private readonly JsonSerializerOptions jsonOptions = new() {
        PropertyNameCaseInsensitive = true,
    };

    public string ResolveDependenciesRootPath(SvcRuntimeOptions options) {
        return ResolveDependenciesRoot(options);
    }

    public SvcModelPaths Resolve(SvcRuntimeOptions options) {
        if (string.IsNullOrWhiteSpace(options.ModelRoot)) {
            throw new ArgumentException("ModelRoot must be set.", nameof(options));
        }

        var dependenciesRoot = ResolveDependenciesRootPath(options);

        var vocoderDirectory = ResolveVocoderDirectory(dependenciesRoot, options.ModelRoot);
        var vocoderConfigPath = ResolveVocoderConfigPath(options, vocoderDirectory);
        var vocoderConfig = LoadVocoderConfig(vocoderConfigPath);
        var exportMetadata = TryLoadExportMetadata(options.ModelRoot);

        var contentEncoderCandidates = ResolveContentEncoderCandidates(options.ContentEncoder, dependenciesRoot, options.ModelRoot);

        return new SvcModelPaths(
            DDspEncoderPath: ResolveOptionalOrFirstExistingFile(
                options.DDspEncoderPath,
                Path.Combine(options.ModelRoot, exportMetadata?.Onnx.Encoder ?? "encoder.onnx"),
                Path.Combine(options.ModelRoot, "encoder.onnx"),
                Path.Combine(options.ModelRoot, "ddsp_encoder.onnx")),
            ReflowVelocityPath: ResolveOptionalOrFirstExistingFile(
                options.ReflowVelocityPath,
                Path.Combine(options.ModelRoot, exportMetadata?.Onnx.Velocity ?? "velocity.onnx"),
                Path.Combine(options.ModelRoot, "velocity.onnx"),
                Path.Combine(options.ModelRoot, "reflow_velocity.onnx")),
            VocoderPath: ResolveOptionalOrFirstExistingFile(
                options.VocoderPath,
                Path.Combine(vocoderDirectory, vocoderConfig.Model)),
            RmvpePath: ResolveOptionalOrFirstExistingFile(
                options.RmvpePath,
                Path.Combine(dependenciesRoot, "rmvpe", "rmvpe.onnx"),
                Path.Combine(dependenciesRoot, "RMVPE", "rmvpe.onnx")),
            ContentEncoderPath: ResolveOptionalOrFirstExistingFile(
                options.ContentEncoderPath,
                contentEncoderCandidates));
    }

    public VocoderConfig LoadVocoderConfig(string configPath) {
        if (!File.Exists(configPath)) {
            throw new FileNotFoundException("Vocoder config not found.", configPath);
        }
        return yamlDeserializer.Deserialize<VocoderConfig>(File.ReadAllText(configPath));
    }

    public SvcExportMetadata? TryLoadExportMetadata(string modelRoot) {
        var metadataPath = Path.Combine(modelRoot, "svc.json");
        if (!File.Exists(metadataPath)) {
            return null;
        }
        return JsonSerializer.Deserialize<SvcExportMetadata>(File.ReadAllText(metadataPath), jsonOptions);
    }

    public SvcModelConfig? TryLoadModelConfig(string modelRoot) {
        foreach (var path in GetModelConfigCandidates(modelRoot)) {
            if (File.Exists(path)) {
                return yamlDeserializer.Deserialize<SvcModelConfig>(File.ReadAllText(path));
            }
        }
        return null;
    }

    private static string ResolveDependenciesRoot(SvcRuntimeOptions options) {
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(options.DependenciesRoot)) {
            candidates.Add(options.DependenciesRoot!);
            candidates.Add(Path.Combine(options.DependenciesRoot!, "Dependencies"));
            candidates.Add(Path.Combine(options.DependenciesRoot!, "dependencies"));
        }
        candidates.Add(Path.Combine(options.ModelRoot, "dependencies"));
        candidates.Add(Path.Combine(options.ModelRoot, "Dependencies"));
        candidates.Add(Path.Combine(Directory.GetParent(options.ModelRoot)?.FullName ?? options.ModelRoot, "Dependencies"));

        foreach (var candidate in candidates.Distinct(StringComparer.Ordinal)) {
            if (Directory.Exists(candidate)) {
                return candidate;
            }
        }

        return candidates.First();
    }

    private static IEnumerable<string> GetModelConfigCandidates(string modelRoot) {
        yield return Path.Combine(modelRoot, "config.yaml");
        var parent = Directory.GetParent(modelRoot);
        if (parent is not null) {
            yield return Path.Combine(parent.FullName, "config.yaml");
        }
    }

    private static string ResolveFirstExistingDirectory(params string[] candidates) {
        var directory = candidates.FirstOrDefault(Directory.Exists);
        return directory ?? throw new DirectoryNotFoundException(
            "None of the candidate directories exist:" + Environment.NewLine + string.Join(Environment.NewLine, candidates));
    }

    private static string ResolveVocoderDirectory(string dependenciesRoot, string modelRoot) {
        var directCandidates = new[] {
            Path.Combine(dependenciesRoot, "pc-nsf-hifigan"),
            Path.Combine(dependenciesRoot, "pc_nsf_hifigan"),
            Path.Combine(modelRoot, "pc-nsf-hifigan"),
            Path.Combine(modelRoot, "pc_nsf_hifigan"),
        };
        var direct = directCandidates.FirstOrDefault(Directory.Exists);
        if (direct is not null) {
            return direct;
        }

        var searchRoots = new[] { dependenciesRoot, modelRoot }
            .Where(Directory.Exists)
            .Distinct(StringComparer.Ordinal);
        foreach (var root in searchRoots) {
            var matched = Directory.EnumerateDirectories(root, "*", SearchOption.TopDirectoryOnly)
                .FirstOrDefault(path => {
                    var name = Path.GetFileName(path);
                    return name.StartsWith("pc_nsf_hifigan", StringComparison.OrdinalIgnoreCase) ||
                           name.StartsWith("pc-nsf-hifigan", StringComparison.OrdinalIgnoreCase);
                });
            if (matched is not null) {
                return matched;
            }
        }

        return ResolveFirstExistingDirectory(directCandidates);
    }

    private static string ResolveVocoderConfigPath(SvcRuntimeOptions options, string fallbackDirectory) {
        if (!string.IsNullOrWhiteSpace(options.VocoderPath)) {
            var explicitPath = Path.GetFullPath(options.VocoderPath!);
            if (Directory.Exists(explicitPath)) {
                return ResolveFirstExistingFile(Path.Combine(explicitPath, "vocoder.yaml"));
            }

            var directory = Path.GetDirectoryName(explicitPath);
            if (!string.IsNullOrWhiteSpace(directory)) {
                return ResolveFirstExistingFile(Path.Combine(directory, "vocoder.yaml"));
            }
        }

        return ResolveFirstExistingFile(Path.Combine(fallbackDirectory, "vocoder.yaml"));
    }

    private static string[] ResolveContentEncoderCandidates(ContentEncoderKind encoderKind, string dependenciesRoot, string modelRoot) {
        var candidates = new List<string>();
        switch (encoderKind) {
            case ContentEncoderKind.ContentVec768L12:
                AddContentVecCandidates(candidates, dependenciesRoot, modelRoot, "contentvec.onnx", "encoder.onnx");
                break;
            case ContentEncoderKind.ContentVec768L12Tta2x:
                AddContentVecCandidates(candidates, dependenciesRoot, modelRoot, "contentvec.onnx", "contentvec_tta2x.onnx", "encoder.onnx");
                break;
            case ContentEncoderKind.HubertSoft:
                candidates.Add(Path.Combine(dependenciesRoot, "hubertsoft", "hubertsoft.onnx"));
                candidates.Add(Path.Combine(modelRoot, "hubertsoft.onnx"));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(encoderKind), encoderKind, null);
        }
        return candidates.Distinct(StringComparer.Ordinal).ToArray();
    }

    private static void AddContentVecCandidates(List<string> candidates, string dependenciesRoot, string modelRoot, params string[] fileNames) {
        foreach (var fileName in fileNames) {
            candidates.Add(Path.Combine(dependenciesRoot, "contentvec", fileName));
            candidates.Add(Path.Combine(modelRoot, "contentvec", fileName));
        }

        var searchRoots = new[] { dependenciesRoot, modelRoot }
            .Where(Directory.Exists)
            .Distinct(StringComparer.Ordinal);
        foreach (var root in searchRoots) {
            foreach (var directory in Directory.EnumerateDirectories(root, "*", SearchOption.TopDirectoryOnly)) {
                var name = Path.GetFileName(directory);
                if (!name.StartsWith("contentvec", StringComparison.OrdinalIgnoreCase) &&
                    !name.StartsWith("content-vec", StringComparison.OrdinalIgnoreCase)) {
                    continue;
                }
                foreach (var fileName in fileNames) {
                    candidates.Add(Path.Combine(directory, fileName));
                }
            }
        }
    }

    private static string ResolveFirstExistingFile(params string[] candidates) {
        var uniqueCandidates = candidates.Distinct(StringComparer.Ordinal).ToArray();
        var path = uniqueCandidates.FirstOrDefault(File.Exists);
        return path ?? throw new FileNotFoundException(
            "None of the candidate files exist:" + Environment.NewLine + string.Join(Environment.NewLine, uniqueCandidates));
    }

    private static string ResolveOptionalOrFirstExistingFile(string? explicitPath, params string[] candidates) {
        if (!string.IsNullOrWhiteSpace(explicitPath)) {
            var fullPath = Path.GetFullPath(explicitPath);
            if (File.Exists(fullPath)) {
                return fullPath;
            }

            throw new FileNotFoundException("Explicitly specified asset file does not exist.", fullPath);
        }

        return ResolveFirstExistingFile(candidates);
    }
}

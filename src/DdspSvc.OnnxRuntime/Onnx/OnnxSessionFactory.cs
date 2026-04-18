using DdspSvc.OnnxRuntime.Models;
using Microsoft.ML.OnnxRuntime;

namespace DdspSvc.OnnxRuntime.Onnx;

public sealed class OnnxSessionFactory {
    private readonly SvcExecutionProvider executionProvider;
    private readonly int executionDeviceId;

    public OnnxSessionFactory()
        : this(SvcExecutionProvider.Cpu, 0) {
    }

    public OnnxSessionFactory(SvcRuntimeOptions options)
        : this(options.ExecutionProvider, options.ExecutionDeviceId) {
    }

    public OnnxSessionFactory(SvcExecutionProvider executionProvider, int executionDeviceId = 0) {
        this.executionProvider = executionProvider;
        this.executionDeviceId = executionDeviceId;
    }

    public InferenceSession Create(string modelPath, OnnxRunnerChoice runnerChoice = OnnxRunnerChoice.Default) {
        return runnerChoice == OnnxRunnerChoice.Cpu
            ? new InferenceSession(modelPath)
            : new InferenceSession(modelPath, BuildSessionOptions());
    }

    public InferenceSession Create(byte[] modelBytes, OnnxRunnerChoice runnerChoice = OnnxRunnerChoice.Default) {
        return runnerChoice == OnnxRunnerChoice.Cpu
            ? new InferenceSession(modelBytes)
            : new InferenceSession(modelBytes, BuildSessionOptions());
    }

    public void VerifyInputNames(InferenceSession session, IEnumerable<NamedOnnxValue> inputs) {
        var sessionInputNames = session.InputNames.ToHashSet(StringComparer.Ordinal);
        var providedInputNames = inputs.Select(input => input.Name).ToHashSet(StringComparer.Ordinal);

        var missing = sessionInputNames.Except(providedInputNames).OrderBy(name => name).ToArray();
        if (missing.Length > 0) {
            throw new ArgumentException($"Missing input(s): {string.Join(", ", missing)}");
        }

        var unexpected = providedInputNames.Except(sessionInputNames).OrderBy(name => name).ToArray();
        if (unexpected.Length > 0) {
            throw new ArgumentException($"Unexpected input(s): {string.Join(", ", unexpected)}");
        }
    }

    private SessionOptions BuildSessionOptions() {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        switch (executionProvider) {
            case SvcExecutionProvider.Cpu:
                options.AppendExecutionProvider_CPU();
                break;
            case SvcExecutionProvider.Cuda:
                options.AppendExecutionProvider_CUDA(executionDeviceId);
                break;
            default:
                throw new NotSupportedException($"Unsupported execution provider '{executionProvider}'.");
        }
        return options;
    }
}

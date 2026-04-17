using Microsoft.ML.OnnxRuntime;

namespace DdspSvc.OnnxRuntime.Onnx;

public sealed class OnnxSessionFactory {
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

    private static SessionOptions BuildSessionOptions() {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        return options;
    }
}

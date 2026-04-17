namespace DdspSvc.OnnxRuntime;

public sealed class UnsupportedSvcFeatureException : NotSupportedException {
    public string FeatureName { get; }

    public UnsupportedSvcFeatureException(string featureName, string message)
        : base(message) {
        FeatureName = featureName;
    }
}

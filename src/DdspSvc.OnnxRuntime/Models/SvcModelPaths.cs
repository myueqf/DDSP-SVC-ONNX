namespace DdspSvc.OnnxRuntime.Models;

public sealed record SvcModelPaths(
    string DDspEncoderPath,
    string ReflowVelocityPath,
    string VocoderPath,
    string RmvpePath,
    string ContentEncoderPath);

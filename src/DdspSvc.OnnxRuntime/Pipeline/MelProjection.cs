namespace DdspSvc.OnnxRuntime.Pipeline;

public static class MelProjection {
    public const float DefaultSpecMax = 2f;
    public const float DefaultSpecMin = -12f;
    private const float Log10E = 0.434294f;

    public static float[] Denormalize(float[] normalizedMel, string? melBase = null) {
        ArgumentNullException.ThrowIfNull(normalizedMel);

        var convertToLog10 = RequiresLog10Projection(melBase);
        var mel = new float[normalizedMel.Length];
        var scale = DefaultSpecMax - DefaultSpecMin;
        for (var i = 0; i < normalizedMel.Length; i++) {
            var value = (normalizedMel[i] + 1f) * 0.5f * scale + DefaultSpecMin;
            mel[i] = convertToLog10 ? value * Log10E : value;
        }
        return mel;
    }

    public static bool RequiresLog10Projection(string? melBase) {
        if (string.IsNullOrWhiteSpace(melBase)) {
            return false;
        }

        return melBase.Trim().ToLowerInvariant() switch {
            "10" => true,
            "log10" => true,
            "e" => false,
            "ln" => false,
            "log" => false,
            _ => throw new NotSupportedException($"Unsupported mel base '{melBase}'."),
        };
    }

    public static float[] ClampNormalized(float[] normalizedMel, float min = -1f, float max = 1f) {
        ArgumentNullException.ThrowIfNull(normalizedMel);
        if (min > max) {
            throw new ArgumentException("min must be less than or equal to max.");
        }

        var clamped = new float[normalizedMel.Length];
        for (var i = 0; i < normalizedMel.Length; i++) {
            clamped[i] = Math.Clamp(normalizedMel[i], min, max);
        }
        return clamped;
    }

    public static float[] FlattenToFrameMajor(float[] mel, int frames, int melBins) {
        ArgumentNullException.ThrowIfNull(mel);
        if (frames <= 0) {
            throw new ArgumentOutOfRangeException(nameof(frames));
        }
        if (melBins <= 0) {
            throw new ArgumentOutOfRangeException(nameof(melBins));
        }
        if (mel.Length != frames * melBins) {
            throw new ArgumentException("Mel tensor size does not match frames * melBins.", nameof(mel));
        }

        var flattened = new float[mel.Length];
        for (var f = 0; f < frames; f++) {
            for (var c = 0; c < melBins; c++) {
                flattened[f * melBins + c] = mel[c * frames + f];
            }
        }
        return flattened;
    }
}

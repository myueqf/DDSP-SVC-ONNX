namespace DdspSvc.OnnxRuntime.Audio;

public static class VolumeMask {
    public static float[] BuildBinary(float[] volume, int hopSize, int targetLength, float thresholdDb = -60f) {
        ArgumentNullException.ThrowIfNull(volume);
        var threshold = MathF.Pow(10f, thresholdDb / 20f);
        var frameMask = new float[volume.Length];
        for (var i = 0; i < volume.Length; i++) {
            frameMask[i] = volume[i] > threshold ? 1f : 0f;
        }
        return UpsampleLikeOriginal(frameMask, hopSize, targetLength);
    }

    public static float[] Build(
        float[] volume,
        int hopSize,
        int targetLength,
        float silenceThresholdDb = -72f,
        float fullVolumeThresholdDb = -48f,
        int smoothingSamples = 1024) {
        ArgumentNullException.ThrowIfNull(volume);
        if (hopSize <= 0) {
            throw new ArgumentOutOfRangeException(nameof(hopSize));
        }
        if (targetLength < 0) {
            throw new ArgumentOutOfRangeException(nameof(targetLength));
        }
        if (targetLength == 0 || volume.Length == 0) {
            return new float[targetLength];
        }
        var frameMask = new float[volume.Length];
        for (var i = 0; i < volume.Length; i++) {
            var db = 20f * MathF.Log10(MathF.Max(volume[i], 1e-8f));
            frameMask[i] = SmoothStep(silenceThresholdDb, fullVolumeThresholdDb, db);
        }
        var mask = UpsampleLikeOriginal(frameMask, hopSize, targetLength);
        if (smoothingSamples > 1) {
            SmoothInPlace(mask, smoothingSamples);
        }
        return mask;
    }

    public static void ApplyInPlace(float[] audio, float[] mask) {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(mask);
        if (audio.Length != mask.Length) {
            throw new ArgumentException("Audio and mask must have the same length.");
        }

        for (var i = 0; i < audio.Length; i++) {
            audio[i] *= mask[i];
        }
    }

    private static float[] UpsampleLikeOriginal(float[] frames, int factor, int targetLength) {
        if (factor <= 0) {
            throw new ArgumentOutOfRangeException(nameof(factor));
        }
        if (targetLength < 0) {
            throw new ArgumentOutOfRangeException(nameof(targetLength));
        }
        if (targetLength == 0 || frames.Length == 0) {
            return new float[targetLength];
        }

        // Match ddsp.core.upsample():
        // 1. append the last frame once
        // 2. linearly interpolate with align_corners=True to ((T + 1) * factor + 1)
        // 3. drop the final sample
        var extendedLength = frames.Length + 1;
        var outputLength = extendedLength * factor;
        var interpolationSize = outputLength + 1;

        var extended = new float[extendedLength];
        Array.Copy(frames, extended, frames.Length);
        extended[^1] = frames[^1];

        var result = new float[targetLength];
        var limit = Math.Min(targetLength, outputLength);
        var maxInputIndex = extendedLength - 1;
        var maxOutputIndex = interpolationSize - 1;
        for (var i = 0; i < limit; i++) {
            var position = i * (double)maxInputIndex / maxOutputIndex;
            var left = Math.Min((int)Math.Floor(position), maxInputIndex - 1);
            var frac = (float)(position - left);
            result[i] = extended[left] + (extended[left + 1] - extended[left]) * frac;
        }

        // If the caller asks for more samples than the original upsample would emit,
        // hold the final value rather than reading past the generated mask.
        for (var i = limit; i < targetLength; i++) {
            result[i] = frames[^1];
        }
        return result;
    }

    private static float SmoothStep(float edge0, float edge1, float value) {
        if (edge0 >= edge1) {
            return value >= edge1 ? 1f : 0f;
        }
        var t = Math.Clamp((value - edge0) / (edge1 - edge0), 0f, 1f);
        return t * t * (3f - 2f * t);
    }

    private static void SmoothInPlace(float[] values, int windowSize) {
        var smoothed = new float[values.Length];
        var halfWindow = Math.Max(1, windowSize / 2);
        for (var i = 0; i < values.Length; i++) {
            var start = Math.Max(0, i - halfWindow);
            var end = Math.Min(values.Length, i + halfWindow + 1);
            float sum = 0f;
            for (var j = start; j < end; j++) {
                sum += values[j];
            }
            smoothed[i] = sum / (end - start);
        }
        Array.Copy(smoothed, values, values.Length);
    }
}

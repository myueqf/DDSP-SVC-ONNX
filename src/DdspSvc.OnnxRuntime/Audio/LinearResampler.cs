namespace DdspSvc.OnnxRuntime.Audio;

public static class LinearResampler {
    public static float[] Resample(ReadOnlySpan<float> samples, int sourceSampleRate, int targetSampleRate) {
        if (sourceSampleRate <= 0) {
            throw new ArgumentOutOfRangeException(nameof(sourceSampleRate));
        }
        if (targetSampleRate <= 0) {
            throw new ArgumentOutOfRangeException(nameof(targetSampleRate));
        }
        if (samples.Length == 0) {
            return [];
        }
        if (sourceSampleRate == targetSampleRate) {
            return samples.ToArray();
        }

        var outputLength = Math.Max(1, (int)Math.Round(samples.Length * (double)targetSampleRate / sourceSampleRate));
        var result = new float[outputLength];
        var scale = (double)sourceSampleRate / targetSampleRate;

        for (var i = 0; i < outputLength; i++) {
            var srcPosition = i * scale;
            var left = (int)Math.Floor(srcPosition);
            var right = Math.Min(left + 1, samples.Length - 1);
            var frac = srcPosition - left;
            var leftValue = samples[Math.Min(left, samples.Length - 1)];
            var rightValue = samples[right];
            result[i] = (float)(leftValue + (rightValue - leftValue) * frac);
        }

        return result;
    }
}

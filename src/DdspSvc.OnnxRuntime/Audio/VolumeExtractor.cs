namespace DdspSvc.OnnxRuntime.Audio;

public sealed class VolumeExtractor {
    private readonly int hopSize;
    private readonly int winSize;

    public VolumeExtractor(int hopSize = 512, int winSize = 2048) {
        if (hopSize <= 0) {
            throw new ArgumentOutOfRangeException(nameof(hopSize));
        }
        if (winSize <= 0) {
            throw new ArgumentOutOfRangeException(nameof(winSize));
        }
        this.hopSize = hopSize;
        this.winSize = winSize;
    }

    public float[] Extract(ReadOnlySpan<float> audio) {
        var padded = ReflectPad(audio, winSize / 2, (winSize + 1) / 2);
        var frameCount = audio.Length / hopSize + 1;
        var result = new float[frameCount];

        for (var frame = 0; frame < frameCount; frame++) {
            var start = frame * hopSize;
            double mean = 0;
            double meanSquare = 0;
            for (var i = 0; i < winSize; i++) {
                var sample = padded[start + i];
                mean += sample;
                meanSquare += sample * sample;
            }
            mean /= winSize;
            meanSquare /= winSize;
            result[frame] = (float)Math.Sqrt(Math.Max(0d, meanSquare - mean * mean));
        }

        return result;
    }

    private static float[] ReflectPad(ReadOnlySpan<float> audio, int left, int right) {
        if (audio.Length == 0) {
            return new float[left + right];
        }

        var result = new float[left + audio.Length + right];
        for (var i = 0; i < left; i++) {
            result[i] = audio[ReflectIndex(i - left, audio.Length)];
        }
        audio.CopyTo(result.AsSpan(left, audio.Length));
        for (var i = 0; i < right; i++) {
            result[left + audio.Length + i] = audio[ReflectIndex(audio.Length + i, audio.Length)];
        }
        return result;
    }

    private static int ReflectIndex(int index, int length) {
        if (length == 1) {
            return 0;
        }
        while (index < 0 || index >= length) {
            index = index < 0 ? -index - 1 : 2 * length - index - 1;
        }
        return index;
    }
}

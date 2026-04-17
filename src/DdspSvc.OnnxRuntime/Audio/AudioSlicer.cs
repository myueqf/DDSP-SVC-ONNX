namespace DdspSvc.OnnxRuntime.Audio;

public sealed class AudioSlicer {
    private readonly float threshold;
    private readonly int hopSize;
    private readonly int winSize;
    private readonly int minLength;
    private readonly int minInterval;
    private readonly int maxSilKept;

    public AudioSlicer(
        int sampleRate,
        float thresholdDb = -40f,
        int minLengthMs = 5000,
        int minIntervalMs = 300,
        int hopSizeMs = 20,
        int maxSilKeptMs = 5000) {
        if (!(minLengthMs >= minIntervalMs && minIntervalMs >= hopSizeMs)) {
            throw new ArgumentException("min_length >= min_interval >= hop_size must hold.");
        }
        if (maxSilKeptMs < hopSizeMs) {
            throw new ArgumentException("max_sil_kept >= hop_size must hold.");
        }

        var minIntervalSamples = sampleRate * minIntervalMs / 1000f;
        threshold = MathF.Pow(10f, thresholdDb / 20f);
        hopSize = (int)MathF.Round(sampleRate * hopSizeMs / 1000f);
        winSize = Math.Min((int)MathF.Round(minIntervalSamples), 4 * hopSize);
        minLength = (int)MathF.Round(sampleRate * minLengthMs / 1000f / hopSize);
        minInterval = (int)MathF.Round(minIntervalSamples / hopSize);
        maxSilKept = (int)MathF.Round(sampleRate * maxSilKeptMs / 1000f / hopSize);
    }

    public IReadOnlyList<(bool IsSilence, int StartSample, int EndSample)> Slice(float[] waveform) {
        if (waveform.Length <= minLength * hopSize) {
            return [(false, 0, waveform.Length)];
        }

        var rmsList = ComputeRms(waveform);
        var silenceTags = new List<(int Start, int End)>();
        int? silenceStart = null;
        var clipStart = 0;

        for (var i = 0; i < rmsList.Length; i++) {
            if (rmsList[i] < threshold) {
                silenceStart ??= i;
                continue;
            }

            if (silenceStart is null) {
                continue;
            }

            var isLeadingSilence = silenceStart == 0 && i > maxSilKept;
            var needSliceMiddle = i - silenceStart >= minInterval && i - clipStart >= minLength;
            if (!isLeadingSilence && !needSliceMiddle) {
                silenceStart = null;
                continue;
            }

            if (i - silenceStart <= maxSilKept) {
                var pos = ArgMin(rmsList, silenceStart.Value, i + 1);
                silenceTags.Add(silenceStart == 0 ? (0, pos) : (pos, pos));
                clipStart = pos;
            } else if (i - silenceStart <= maxSilKept * 2) {
                var pos = ArgMin(rmsList, i - maxSilKept, silenceStart.Value + maxSilKept + 1);
                var posL = ArgMin(rmsList, silenceStart.Value, silenceStart.Value + maxSilKept + 1);
                var posR = ArgMin(rmsList, i - maxSilKept, i + 1);
                if (silenceStart == 0) {
                    silenceTags.Add((0, posR));
                    clipStart = posR;
                } else {
                    silenceTags.Add((Math.Min(posL, pos), Math.Max(posR, pos)));
                    clipStart = Math.Max(posR, pos);
                }
            } else {
                var posL = ArgMin(rmsList, silenceStart.Value, silenceStart.Value + maxSilKept + 1);
                var posR = ArgMin(rmsList, i - maxSilKept, i + 1);
                silenceTags.Add(silenceStart == 0 ? (0, posR) : (posL, posR));
                clipStart = posR;
            }

            silenceStart = null;
        }

        var totalFrames = rmsList.Length;
        if (silenceStart is not null && totalFrames - silenceStart >= minInterval) {
            var silenceEnd = Math.Min(totalFrames, silenceStart.Value + maxSilKept);
            var pos = ArgMin(rmsList, silenceStart.Value, silenceEnd + 1);
            silenceTags.Add((pos, totalFrames + 1));
        }

        if (silenceTags.Count == 0) {
            return [(false, 0, waveform.Length)];
        }

        var chunks = new List<(bool IsSilence, int StartSample, int EndSample)>();
        if (silenceTags[0].Start != 0) {
            chunks.Add((false, 0, Math.Min(waveform.Length, silenceTags[0].Start * hopSize)));
        }
        for (var i = 0; i < silenceTags.Count; i++) {
            if (i > 0) {
                chunks.Add((
                    false,
                    silenceTags[i - 1].End * hopSize,
                    Math.Min(waveform.Length, silenceTags[i].Start * hopSize)));
            }
            chunks.Add((
                true,
                silenceTags[i].Start * hopSize,
                Math.Min(waveform.Length, silenceTags[i].End * hopSize)));
        }
        if (silenceTags[^1].End * hopSize < waveform.Length) {
            chunks.Add((false, silenceTags[^1].End * hopSize, waveform.Length));
        }
        return chunks;
    }

    private float[] ComputeRms(float[] waveform) {
        if (waveform.Length == 0) {
            return [0f];
        }

        var padded = ReflectPad(waveform, winSize / 2, winSize / 2);
        var frameCount = waveform.Length / hopSize + 1;
        var rms = new float[frameCount];
        for (var frame = 0; frame < frameCount; frame++) {
            var start = frame * hopSize;
            double sumSquares = 0;
            for (var i = 0; i < winSize; i++) {
                var sample = padded[start + i];
                sumSquares += sample * sample;
            }
            rms[frame] = (float)Math.Sqrt(sumSquares / winSize);
        }
        return rms;
    }

    private static float[] ReflectPad(ReadOnlySpan<float> audio, int left, int right) {
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

    private static int ArgMin(float[] values, int startInclusive, int endExclusive) {
        var start = Math.Max(0, startInclusive);
        var end = Math.Min(values.Length, endExclusive);
        var bestIndex = start;
        var bestValue = values[start];
        for (var i = start + 1; i < end; i++) {
            if (values[i] < bestValue) {
                bestValue = values[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}

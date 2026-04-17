namespace DdspSvc.OnnxRuntime.Pipeline;

public static class PitchConditioning {
    public static float[] InterpolateUnvoiced(float[] f0, float minF0Hz = 65f) {
        ArgumentNullException.ThrowIfNull(f0);
        if (minF0Hz <= 0f) {
            throw new ArgumentOutOfRangeException(nameof(minF0Hz));
        }

        var conditioned = (float[])f0.Clone();
        var firstVoiced = Array.FindIndex(conditioned, value => value > 0f && !float.IsNaN(value));
        if (firstVoiced < 0) {
            Array.Fill(conditioned, minF0Hz);
            return conditioned;
        }

        for (var i = 0; i < firstVoiced; i++) {
            conditioned[i] = conditioned[firstVoiced];
        }

        var previousVoiced = firstVoiced;
        var index = firstVoiced + 1;
        while (index < conditioned.Length) {
            if (conditioned[index] > 0f && !float.IsNaN(conditioned[index])) {
                previousVoiced = index;
                index++;
                continue;
            }

            var gapStart = index;
            while (index < conditioned.Length && !(conditioned[index] > 0f && !float.IsNaN(conditioned[index]))) {
                index++;
            }

            if (index < conditioned.Length) {
                var left = conditioned[previousVoiced];
                var right = conditioned[index];
                var gapLength = index - previousVoiced;
                for (var i = 1; i < gapLength; i++) {
                    var ratio = (float)i / gapLength;
                    conditioned[previousVoiced + i] = left + (right - left) * ratio;
                }
                previousVoiced = index;
            } else {
                for (var i = gapStart; i < conditioned.Length; i++) {
                    conditioned[i] = conditioned[previousVoiced];
                }
            }
        }

        for (var i = 0; i < conditioned.Length; i++) {
            if (!(conditioned[i] >= minF0Hz) || float.IsNaN(conditioned[i])) {
                conditioned[i] = minF0Hz;
            }
        }
        return conditioned;
    }
}

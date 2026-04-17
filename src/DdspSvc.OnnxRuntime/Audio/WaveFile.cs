using System.Text;

namespace DdspSvc.OnnxRuntime.Audio;

public sealed class WaveFile {
    public required int SampleRate { get; init; }
    public required int Channels { get; init; }
    public required float[] Samples { get; init; }

    public static WaveFile ReadMono16(string path) {
        using var reader = OpenReader(path);
        var result = reader.ReadMonoSamples((int)reader.TotalSamples);
        return new WaveFile {
            SampleRate = reader.SampleRate,
            Channels = 1,
            Samples = result,
        };
    }

    public static WaveFileReader OpenReader(string path) => new(path);

    public static WaveFileWriter OpenWriter(string path, int sampleRate) => new(path, sampleRate);

    public static void WriteMono16(string path, float[] samples, int sampleRate) {
        using var writer = OpenWriter(path, sampleRate);
        writer.WriteMonoSamples(samples);
    }

    internal static WaveFormatInfo ReadFormat(BinaryReader reader, Stream stream) {
        var riff = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (riff != "RIFF") {
            throw new InvalidDataException("Invalid WAV file: missing RIFF header.");
        }
        reader.ReadInt32();
        var wave = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (wave != "WAVE") {
            throw new InvalidDataException("Invalid WAV file: missing WAVE header.");
        }

        short channels = 0;
        int sampleRate = 0;
        short bitsPerSample = 0;
        long dataOffset = -1;
        int dataSize = 0;

        while (stream.Position < stream.Length) {
            var chunkIdBytes = reader.ReadBytes(4);
            if (chunkIdBytes.Length < 4) {
                break;
            }
            var chunkId = Encoding.ASCII.GetString(chunkIdBytes);
            var chunkSize = reader.ReadInt32();
            if (chunkId == "fmt ") {
                var audioFormat = reader.ReadInt16();
                channels = reader.ReadInt16();
                sampleRate = reader.ReadInt32();
                reader.ReadInt32();
                reader.ReadInt16();
                bitsPerSample = reader.ReadInt16();
                if (chunkSize > 16) {
                    reader.ReadBytes(chunkSize - 16);
                }
                if (audioFormat != 1) {
                    throw new NotSupportedException("Only PCM WAV is supported.");
                }
            } else if (chunkId == "data") {
                dataOffset = stream.Position;
                dataSize = chunkSize;
                stream.Position += chunkSize;
            } else {
                stream.Position += chunkSize;
            }

            if ((chunkSize & 1) != 0 && stream.Position < stream.Length) {
                stream.Position += 1;
            }
        }

        if (dataOffset < 0) {
            throw new InvalidDataException("Invalid WAV file: missing data chunk.");
        }
        if (channels <= 0 || sampleRate <= 0 || bitsPerSample != 16) {
            throw new NotSupportedException("Only 16-bit PCM WAV is supported.");
        }

        return new WaveFormatInfo(sampleRate, channels, bitsPerSample, dataOffset, dataSize);
    }
}

public sealed class WaveFileReader : IDisposable {
    private readonly FileStream stream;
    private readonly BinaryReader reader;
    private readonly long dataOffset;
    private readonly long totalFrames;
    private long framesRead;
    private bool disposed;

    public int SampleRate { get; }
    public int Channels { get; }
    public long TotalSamples => totalFrames;

    public WaveFileReader(string path) {
        stream = File.OpenRead(path);
        reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        var format = WaveFile.ReadFormat(reader, stream);
        SampleRate = format.SampleRate;
        Channels = format.Channels;
        dataOffset = format.DataOffset;
        totalFrames = format.DataSize / 2 / format.Channels;
        stream.Position = dataOffset;
    }

    public float[] ReadMonoSamples(int sampleCount) {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (sampleCount < 0) {
            throw new ArgumentOutOfRangeException(nameof(sampleCount));
        }
        var remaining = Math.Max(0, totalFrames - framesRead);
        var framesToRead = (int)Math.Min(sampleCount, remaining);
        if (framesToRead == 0) {
            return [];
        }

        var result = new float[framesToRead];
        for (var i = 0; i < framesToRead; i++) {
            int sum = 0;
            for (var ch = 0; ch < Channels; ch++) {
                sum += reader.ReadInt16();
            }
            result[i] = sum / (float)Channels / 32768f;
        }
        framesRead += framesToRead;
        return result;
    }

    public void Dispose() {
        if (disposed) {
            return;
        }
        reader.Dispose();
        stream.Dispose();
        disposed = true;
    }
}

public sealed class WaveFileWriter : IDisposable {
    private readonly FileStream stream;
    private readonly BinaryWriter writer;
    private readonly int sampleRate;
    private long samplesWritten;
    private bool disposed;

    public WaveFileWriter(string path, int sampleRate) {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        this.sampleRate = sampleRate;
        stream = File.Create(path);
        writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        WriteHeader(dataSize: 0);
    }

    public void WriteMonoSamples(ReadOnlySpan<float> samples) {
        ObjectDisposedException.ThrowIf(disposed, this);
        foreach (var sample in samples) {
            var clamped = Math.Clamp(sample, -1f, 1f);
            writer.Write((short)Math.Round(clamped * 32767f));
        }
        samplesWritten += samples.Length;
    }

    public void Dispose() {
        if (disposed) {
            return;
        }
        writer.Flush();
        stream.Position = 0;
        WriteHeader(checked((int)(samplesWritten * 2)));
        writer.Dispose();
        stream.Dispose();
        disposed = true;
    }

    private void WriteHeader(int dataSize) {
        stream.Position = 0;
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataSize);
    }
}

internal sealed record WaveFormatInfo(
    int SampleRate,
    int Channels,
    short BitsPerSample,
    long DataOffset,
    int DataSize);

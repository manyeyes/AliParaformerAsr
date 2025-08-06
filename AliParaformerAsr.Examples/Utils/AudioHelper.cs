using NAudio.Wave;
using System.Diagnostics;

/// <summary>
/// audio processing
/// Copyright (c)  2023 by manyeyes
/// </summary>
namespace AliParaformerAsr.Examples.Utils
{
    public class AudioHelper
    {
        /// <summary>
        /// get file samples
        /// supports Windows, Mac, and Linux
        /// </summary>
        /// <param name="wavFilePath"></param>
        /// <param name="duration"></param>
        /// <returns></returns>
        public static float[]? GetFileSamples(string wavFilePath, ref TimeSpan duration)
        {
            float[]? wavdata = null;
            if (!File.Exists(wavFilePath))
            {
                Trace.Assert(File.Exists(wavFilePath), "file does not exist:" + wavFilePath);
                return wavdata;
            }
            AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
            byte[] datas = new byte[_audioFileReader.Length];
            _audioFileReader.Read(datas, 0, datas.Length);
            duration = _audioFileReader.TotalTime;
            wavdata = new float[datas.Length / sizeof(float)];
            Buffer.BlockCopy(datas, 0, wavdata, 0, datas.Length);
            return wavdata;
        }

        /// <summary>
        /// get media sample
        /// supports Windows only
        /// </summary>
        /// <param name="mediaFilePath"></param>
        /// <param name="duration"></param>
        /// <returns></returns>
        public static float[]? GetMediaSample(string mediaFilePath, ref TimeSpan duration)
        {
            float[]? wavdata = null;
            try
            {
                if (!File.Exists(mediaFilePath))
                {
                    Trace.Assert(File.Exists(mediaFilePath), "file does not exist:" + mediaFilePath);
                    return wavdata;
                }
                using (MediaFoundationReader _mediaFileReader = new MediaFoundationReader(mediaFilePath))
                {
                    WaveFormat OLDfmt = _mediaFileReader.WaveFormat;
                    int channels = 1;
                    int bitsPerSample = 32;
                    int newSampleRate = 16000;
                    var targetFormat = new WaveFormat(newSampleRate, bitsPerSample, channels);
                    using (var _mediaFileReaderResampler = new MediaFoundationResampler(_mediaFileReader, targetFormat))
                    {
                        _mediaFileReaderResampler.ResamplerQuality = 60; // 设置重采样质量 (0-100)
                        int bytesPerFrame = _mediaFileReaderResampler.WaveFormat.BitsPerSample / 8 * OLDfmt.Channels;
                        int bufferedFrames = (int)Math.Ceiling(_mediaFileReader.Length / bytesPerFrame
                            * ((float)targetFormat.SampleRate / (float)OLDfmt.SampleRate)
                            * ((float)targetFormat.Channels / OLDfmt.Channels)
                            * (targetFormat.BitsPerSample / OLDfmt.BitsPerSample));
                        ISampleProvider? samples = _mediaFileReaderResampler.ToSampleProvider();
                        wavdata = new float[bufferedFrames * bytesPerFrame / sizeof(float)];
                        samples.Read(wavdata, 0, wavdata.Length);
                        duration = _mediaFileReader.TotalTime;
                    }
                }
            }
            catch (Exception ex)
            {
                //
            }
            return wavdata;
        }
        /// <summary>
        /// get file chunk samples
        /// supports Windows, Mac, and Linux
        /// </summary>
        /// <param name="wavFilePath"></param>
        /// <param name="duration"></param>
        /// <param name="chunkSize"></param>
        /// <returns></returns>
        public static List<float[]> GetFileChunkSamples(string wavFilePath, ref TimeSpan duration, int chunkSize = 160 * 6 * 10)
        {
            List<float[]> wavdatas = new List<float[]>();
            if (!File.Exists(wavFilePath))
            {
                Trace.Assert(File.Exists(wavFilePath), "file does not exist:" + wavFilePath);
                wavdatas.Add(new float[1]);
                return wavdatas;
            }
            AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
            byte[] datas = new byte[_audioFileReader.Length];
            _audioFileReader.Read(datas);
            duration = _audioFileReader.TotalTime;
            float[] wavsdata = new float[datas.Length / sizeof(float)];
            int wavsLength = wavsdata.Length;
            Buffer.BlockCopy(datas, 0, wavsdata, 0, datas.Length);
            int chunkNum = (int)Math.Ceiling((double)wavsLength / chunkSize);
            for (int i = 0; i < chunkNum; i++)
            {
                int offset;
                int dataCount;
                if (Math.Abs(wavsLength - i * chunkSize) > chunkSize)
                {
                    offset = i * chunkSize;
                    dataCount = chunkSize;
                }
                else
                {
                    offset = i * chunkSize;
                    dataCount = wavsLength - i * chunkSize;
                }
                float[] wavdata = new float[dataCount];
                Array.Copy(wavsdata, offset, wavdata, 0, dataCount);
                wavdatas.Add(wavdata);
            }
            return wavdatas;
        }
        /// <summary>
        /// get media chunk sample
        /// supports Windows only
        /// </summary>
        /// <param name="mediaFilePath"></param>
        /// <param name="duration"></param>
        /// <param name="chunkSize"></param>
        /// <returns></returns>
        public static List<float[]> GetMediaChunkSamples(string mediaFilePath, ref TimeSpan duration, int chunkSize = 160 * 6 * 10)
        {
            List<float[]> wavdatas = new List<float[]>();
            List<TimeSpan> durations = new List<TimeSpan>();
            if (!File.Exists(mediaFilePath))
            {
                Trace.Assert(File.Exists(mediaFilePath), "file does not exist:" + mediaFilePath);
                wavdatas.Add(new float[1]);
                return wavdatas;
            }
            using (MediaFoundationReader _mediaFileReader = new MediaFoundationReader(mediaFilePath))
            {
                var OLDfmt = _mediaFileReader.WaveFormat;
                var ieeeFloatWaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(16000, 1); // mono
                using (var _mediaFileReaderResampler = new MediaFoundationResampler(_mediaFileReader, ieeeFloatWaveFormat))
                {
                    int bytesPerFrame = _mediaFileReader.WaveFormat.BitsPerSample / 8 * _mediaFileReader.WaveFormat.Channels;
                    int bufferedFrames = (int)Math.Ceiling(_mediaFileReader.Length / bytesPerFrame * ((float)ieeeFloatWaveFormat.SampleRate / (float)OLDfmt.SampleRate / OLDfmt.Channels));
                    ISampleProvider? _samples = _mediaFileReaderResampler.ToSampleProvider();
                    int chunkNum = (int)Math.Floor((double)bufferedFrames / chunkSize);
                    for (int i = 0; i < chunkNum; i++)
                    {
                        int offset = 0;
                        int dataCount = 0;
                        if (Math.Abs(bufferedFrames - i * chunkSize) > chunkSize)
                        {
                            offset = i * chunkSize;
                            dataCount = chunkSize;
                        }
                        else
                        {
                            offset = i * chunkSize;
                            dataCount = bufferedFrames - i * chunkSize;
                        }
                        float[] _frames = new float[dataCount];
                        if (i >= chunkNum - 1)
                        {
                            _frames = new float[chunkSize];
                        }
                        _samples.Read(_frames, 0, dataCount);
                        TimeSpan curDuration = TimeSpan.FromMilliseconds(dataCount / OLDfmt.Channels * 1000 / 16000);
                        durations.Add(curDuration);
                        wavdatas.Add(_frames);
                    }
                }
            }
            duration = durations.Aggregate(TimeSpan.Zero, (currentTotal, nextDuration) => currentTotal + nextDuration);
            return wavdatas;
        }
        /// <summary>
        /// 通过文件头特征判断是否为音频文件
        /// </summary>
        public static bool IsAudioByHeader(string filePath)
        {
            if (!File.Exists(filePath))
                return false;

            // 读取文件头前16字节（足够判断常见音频类型）
            byte[] header = new byte[16];
            using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                int bytesRead = stream.Read(header, 0, header.Length);
                if (bytesRead < header.Length)
                    return false; // 文件过小，无法判断
            }

            // 检查是否为常见音频文件的文件头
            return IsMp3Header(header) ||
                   IsWavHeader(header) ||
                   IsOggHeader(header) ||
                   IsAviHeader(header) ||
                   IsMp4Header(header) ||
                   IsFlacHeader(header);
        }

        // 判断是否为MP3文件头（ID3v2或帧头）
        private static bool IsMp3Header(byte[] header)
        {
            // ID3v2标记：前3字节为 'I','D','3'
            if (header[0] == 0x49 && header[1] == 0x44 && header[2] == 0x33)
                return true;

            // MP3帧头特征（简化判断）
            // 帧头前11位为11111111111，第12位为0或1
            if (header.Length >= 4)
            {
                byte b1 = header[0];
                byte b2 = header[1];
                bool isFrameHeader = (b1 == 0xFF) && ((b2 & 0xE0) == 0xE0); // 0xE0 = 11100000
                return isFrameHeader;
            }
            return false;
        }

        // 判断是否为WAV文件头（RIFF + WAVE）
        private static bool IsWavHeader(byte[] header)
        {
            // WAV文件头前8字节：'R','I','F','F', 长度, 'W','A','V','E'
            if (header.Length >= 12)
            {
                bool isRiff = header[0] == 0x52 && header[1] == 0x49 &&
                              header[2] == 0x46 && header[3] == 0x46; // "RIFF"
                bool isWave = header[8] == 0x57 && header[9] == 0x41 &&
                              header[10] == 0x56 && header[11] == 0x45; // "WAVE"
                return isRiff && isWave;
            }
            return false;
        }

        // 判断是否为OGG文件头（OggS）
        private static bool IsOggHeader(byte[] header)
        {
            // OGG前4字节：'O','g','g','S'
            return header[0] == 0x4F && header[1] == 0x67 &&
                   header[2] == 0x67 && header[3] == 0x53;
        }

        // 判断是否为FLAC文件头（fLaC）
        private static bool IsFlacHeader(byte[] header)
        {
            // FLAC前4字节：'f','L','a','C'
            return header[0] == 0x66 && header[1] == 0x4C &&
                   header[2] == 0x61 && header[3] == 0x43;
        }
        /// <summary>
        /// 判断是否为MP4文件头
        /// MP4文件通常以"ftyp"作为文件标识
        /// </summary>
        public static bool IsMp4Header(byte[] header)
        {
            // 1. 处理null和长度不足的情况
            if (header == null || header.Length < 8) // 至少需要8字节（4字节大小 + 4字节"ftyp"）
                return false;

            // 2. 检查"ftyp"原子标识（第4-7字节，因为前4字节是原子大小）
            // 注意：标准"ftyp"原子通常位于文件开头，即前8字节为 [大小][ftyp]
            bool isFtypAtom = (header[4] == 0x66 && header[5] == 0x74 &&
                               header[6] == 0x79 && header[7] == 0x70);

            // 3. 兼容部分以"ftyp"直接开头的非标准文件（前4字节即为"ftyp"）
            bool isFtypAtStart = (header[0] == 0x66 && header[1] == 0x74 &&
                                  header[2] == 0x79 && header[3] == 0x70);

            return isFtypAtom || isFtypAtStart;
        }

        /// <summary>
        /// 判断是否为AVI文件头
        /// AVI文件以"RIFF"开头，且后续包含"AVI "标识
        /// </summary>
        public static bool IsAviHeader(byte[] header)
        {
            // AVI文件头特征：
            // 1. 前4字节为 "RIFF"（0x52494646）
            // 2. 第8-11字节为 "AVI "（0x41564920，注意末尾有空格）
            if (header.Length >= 12)
            {
                bool isRiff = header[0] == 0x52 &&  // 'R'
                              header[1] == 0x49 &&  // 'I'
                              header[2] == 0x46 &&  // 'F'
                              header[3] == 0x46;    // 'F'

                bool isAvi = header[8] == 0x41 &&   // 'A'
                             header[9] == 0x56 &&   // 'V'
                             header[10] == 0x49 &&  // 'I'
                             header[11] == 0x20;    // ' '（空格）

                return isRiff && isAvi;
            }
            return false;
        }
    }
}

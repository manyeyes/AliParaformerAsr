using NAudio.Wave;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

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
                        TimeSpan curDuration = TimeSpan.FromMilliseconds(dataCount / 2 * 1000 / 16000);
                        durations.Add(curDuration);
                        wavdatas.Add(_frames);
                    }
                }
            }
            duration = durations.Aggregate(TimeSpan.Zero, (currentTotal, nextDuration) => currentTotal + nextDuration);
            return wavdatas;
        }
    }
}

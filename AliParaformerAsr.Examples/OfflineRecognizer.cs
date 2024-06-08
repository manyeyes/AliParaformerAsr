using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliParaformerAsr.Examples
{
    internal static partial class Program
    {
        public static OfflineRecognizer initOfflineRecognizer(string modelName)
        {
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            string modelFilePath = applicationBase + "./" + modelName + "/model.ts.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
            string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds_init = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            Console.WriteLine("loading_the_model_elapsed_milliseconds:{0}", elapsed_milliseconds_init.ToString());
            return offlineRecognizer;
        }
        public static void OfflineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
            OfflineRecognizer offlineRecognizer = initOfflineRecognizer(modelName);
            TimeSpan total_duration = new TimeSpan(0L);
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int i = 1; i < 4; i++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/example/{0}.wav", i.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        break;
                    }
                    AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
                    byte[] datas = new byte[_audioFileReader.Length];
                    _audioFileReader.Read(datas, 0, datas.Length);
                    TimeSpan duration = _audioFileReader.TotalTime;
                    float[] sample = new float[datas.Length / sizeof(float)];
                    Buffer.BlockCopy(datas, 0, sample, 0, datas.Length);
                    samples.Add(sample);
                    total_duration += duration;
                }
            }
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
            foreach (var sample in samples)
            {
                AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                stream.AddSamples(sample);
                streams.Add(stream);
            }
            List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
            foreach (AliParaformerAsr.Model.OfflineRecognizerResultEntity result in results)
            {
                Console.WriteLine(result.Text);
                //test_AliCTTransformerPunc(result.Text);
                for (int i = 0; i < result.Tokens.Count; i++)
                {
                    Console.WriteLine(string.Format("{0}:[{1},{2}]", result.Tokens[i], result.Timestamps[i].First(), result.Timestamps[i].Last()));
                }

            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("Hello, World!");
        }
    }
}

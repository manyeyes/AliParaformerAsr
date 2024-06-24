using AliParaformerAsr.Examples.Utils;

namespace AliParaformerAsr.Examples
{
    internal static partial class Program
    {
        public static OnlineRecognizer initOnlineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
            string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
            return onlineRecognizer;
        }

        public static void OnlineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx";
            OnlineRecognizer onlineRecognizer = initOnlineRecognizer(modelName);
            TimeSpan total_duration = TimeSpan.Zero;
            TimeSpan start_time = TimeSpan.Zero;
            TimeSpan end_time = TimeSpan.Zero;
            start_time = new TimeSpan(DateTime.Now.Ticks);

            List<List<float[]>> samplesList = new List<List<float[]>>();
            int batchSize = 1;
            int startIndex = 1;
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int n = startIndex; n < startIndex + batchSize; n++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/example/{0}.wav", n.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration);
                    for (int j = 0; j < 10; j++)
                    {
                        samples.Add(new float[400]);
                    }
                    samplesList.Add(samples);
                    total_duration += duration;
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            List<AliParaformerAsr.OnlineStream> onlineStreams = new List<AliParaformerAsr.OnlineStream>();
            List<bool> isEndpoints = new List<bool>();
            List<bool> isEnds = new List<bool>();
            for (int num = 0; num < samplesList.Count; num++)
            {
                AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                onlineStreams.Add(stream);
                isEndpoints.Add(false);
                isEnds.Add(false);
            }
            int i = 0;
            List<AliParaformerAsr.OnlineStream> streams = new List<AliParaformerAsr.OnlineStream>();

            while (true)
            {
                streams = new List<AliParaformerAsr.OnlineStream>();

                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (samplesList[j].Count > i && samplesList.Count > j)
                    {
                        onlineStreams[j].AddSamples(samplesList[j][i]);
                        streams.Add(onlineStreams[j]);
                        isEndpoints[j] = false;
                    }
                    else
                    {
                        streams.Add(onlineStreams[j]);
                        samplesList.Remove(samplesList[j]);
                        isEndpoints[j] = true;
                    }
                }    
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (isEndpoints[j])
                    {
                        if (onlineStreams[j].IsFinished(isEndpoints[j]))
                        {
                            isEnds[j] = true;
                        }
                        else
                        {
                            streams.Add(onlineStreams[j]);
                        }
                    }
                }
                List<AliParaformerAsr.OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
                foreach (AliParaformerAsr.OnlineRecognizerResultEntity result in results_batch)
                {
                    Console.WriteLine(result.text);
                }
                Console.WriteLine("");
                i++;
                bool isAllFinish = true;
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (!isEnds[j])
                    {
                        isAllFinish = false;
                        break;
                    }
                }
                if (isAllFinish)
                {
                    break;
                }
            }
            end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("Hello, World!");
        }        
    }
}

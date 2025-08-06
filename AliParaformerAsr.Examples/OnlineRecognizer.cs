using AliParaformerAsr.Examples.Utils;

namespace AliParaformerAsr.Examples
{
    internal static partial class Program
    {
        private static AliParaformerAsr.OnlineRecognizer? _onlineRecognizer;
        public static OnlineRecognizer? initOnlineRecognizer(string modelName, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_onlineRecognizer == null)
            {
                if (string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                try
                {
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
                    string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
                    string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
                    string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
                    string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
                    _onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath, threadsNum: threadsNum);
                    TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds_init = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                    Console.WriteLine("init_models_elapsed_milliseconds:{0}", elapsed_milliseconds_init.ToString());
                }
                catch (UnauthorizedAccessException)
                {
                    Console.WriteLine("错误：没有访问该文件夹的权限");
                }
                catch (PathTooLongException)
                {
                    Console.WriteLine("错误：文件路径过长");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"发生错误：{ex.Message}");
                }
            }
            return _onlineRecognizer;
        }

        public static void OnlineRecognizer(string streamDecodeMethod = "one", string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx", string modelAccuracy = "int8", int threadsNum = 2, string[]? mediaFilePaths = null)
        {
            OnlineRecognizer? onlineRecognizer = initOnlineRecognizer(modelName);
            if (onlineRecognizer == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            TimeSpan total_duration = TimeSpan.Zero;
            TimeSpan start_time = TimeSpan.Zero;
            TimeSpan end_time = TimeSpan.Zero;
            start_time = new TimeSpan(DateTime.Now.Ticks);

            List<List<float[]>> samplesList = new List<List<float[]>>();
            int batchSize = 2;
            int startIndex = 0;
            int n = 0;
            List<float[]>? samples = new List<float[]>();
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                mediaFilePaths = Directory.GetFiles(Path.Join(applicationBase, modelName, "test_wavs"));
            }
            foreach (string mediaFilePath in mediaFilePaths)
            {
                if (n < startIndex)
                {
                    continue;
                }
                if (batchSize <= n-startIndex)
                {
                    break;
                }
                if (!File.Exists(mediaFilePath))
                {
                    continue;
                }
                if (AudioHelper.IsAudioByHeader(mediaFilePath))
                {
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetMediaChunkSamples(mediaFilePath, ref duration);
                    for (int j = 0; j < 30; j++)
                    {
                        samples.Add(new float[400]);
                    }
                    samplesList.Add(samples);
                    total_duration += duration;
                }
                n++;
            }
            if (samplesList.Count == 0)
            {
                Console.WriteLine("No media file is read!");
                return;
            }
            streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "multi" : streamDecodeMethod;//one ,multi
            if (streamDecodeMethod == "one")
            {
                //one stream decode
                for (int j = 0; j < samplesList.Count; j++)
                {
                    AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                    foreach (float[] samplesItem in samplesList[j])
                    {
                        stream.AddSamples(samplesItem);
                        AliParaformerAsr.Model.OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
                        Console.WriteLine(result.text);
                    }
                }
                // one stream decode
            }
            //if (streamDecodeMethod == "multi")
            //{
            //    //multi stream decode
            //    List<AliParaformerAsr.OnlineStream> onlineStreams = new List<AliParaformerAsr.OnlineStream>();
            //    List<bool> isEndpoints = new List<bool>();
            //    List<bool> isEnds = new List<bool>();
            //    for (int num = 0; num < samplesList.Count; num++)
            //    {
            //        AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
            //        onlineStreams.Add(stream);
            //        isEndpoints.Add(false);
            //        isEnds.Add(false);
            //    }
            //    int i = 0;
            //    List<AliParaformerAsr.OnlineStream> streams = new List<AliParaformerAsr.OnlineStream>();

            //    while (true)
            //    {
            //        streams = new List<AliParaformerAsr.OnlineStream>();

            //        for (int j = 0; j < samplesList.Count; j++)
            //        {
            //            if (samplesList[j].Count > i && samplesList.Count > j)
            //            {
            //                onlineStreams[j].AddSamples(samplesList[j][i]);
            //                streams.Add(onlineStreams[j]);
            //                isEndpoints[j] = false;
            //            }
            //            else
            //            {
            //                streams.Add(onlineStreams[j]);
            //                samplesList.Remove(samplesList[j]);
            //                isEndpoints[j] = true;
            //            }
            //        }
            //        for (int j = 0; j < samplesList.Count; j++)
            //        {
            //            if (isEndpoints[j])
            //            {
            //                if (onlineStreams[j].IsFinished(isEndpoints[j]))
            //                {
            //                    isEnds[j] = true;
            //                }
            //                else
            //                {
            //                    streams.Add(onlineStreams[j]);
            //                }
            //            }
            //        }
            //        List<AliParaformerAsr.Model.OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
            //        foreach (AliParaformerAsr.Model.OnlineRecognizerResultEntity result in results_batch)
            //        {
            //            Console.WriteLine(result.text);
            //        }
            //        Console.WriteLine("");
            //        i++;
            //        bool isAllFinish = true;
            //        for (int j = 0; j < samplesList.Count; j++)
            //        {
            //            if (!isEnds[j])
            //            {
            //                isAllFinish = false;
            //                break;
            //            }
            //        }
            //        if (isAllFinish)
            //        {
            //            break;
            //        }
            //    }
            //}
            if (_onlineRecognizer != null)
            {
                _onlineRecognizer.Dispose();
                _onlineRecognizer = null;
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

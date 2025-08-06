using AliParaformerAsr.Examples.Utils;

namespace AliParaformerAsr.Examples
{
    internal static partial class Program
    {
        private static AliParaformerAsr.OfflineRecognizer? _offlineRecognizer;
        public static OfflineRecognizer InitOfflineRecognizer(string modelName, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_offlineRecognizer == null)
            {
                if (string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                try
                {
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
                    string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
                    string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
                    string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
                    string modelebFilePath = applicationBase + "./" + modelName + "/model_eb.int8.onnx";
                    string hotwordFilePath = applicationBase + "./" + modelName + "/hotword.txt";
                    _offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, modelebFilePath: modelebFilePath, hotwordFilePath: hotwordFilePath, threadsNum: threadsNum);
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
            return _offlineRecognizer;
        }
        public static void OfflineRecognizer(string streamDecodeMethod = "one", string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline", string modelAccuracy = "int8", int threadsNum = 2, string[]? mediaFilePaths = null)
        {
            OfflineRecognizer offlineRecognizer = InitOfflineRecognizer(modelName, modelAccuracy, threadsNum);
            if (offlineRecognizer == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            TimeSpan total_duration = new TimeSpan(0L);
            List<float[]>? samples = new List<float[]>();
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                mediaFilePaths = Directory.GetFiles(Path.Join(applicationBase, modelName, "test_wavs"));
            }
            foreach (string mediaFilePath in mediaFilePaths)
            {
                if (!File.Exists(mediaFilePath))
                {
                    continue;
                }
                if (AudioHelper.IsAudioByHeader(mediaFilePath))
                {
                    TimeSpan duration = TimeSpan.Zero;
                    float[] sample = AudioHelper.GetMediaSample(mediaFilePath: mediaFilePath, duration: ref duration);
                    samples.Add(sample);
                    total_duration += duration;
                }
            }
            if (samples.Count == 0)
            {
                Console.WriteLine("No media file is read!");
                return;
            }
            Console.WriteLine("Automatic speech recognition in progress!");
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "multi" : streamDecodeMethod;//one ,multi
            if (streamDecodeMethod == "one")
            {
                // Non batch method
                Console.WriteLine("Recognition results:\r\n");
                foreach (var sample in samples)
                {
                    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                    stream.AddSamples(sample);
                    AliParaformerAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                    Console.WriteLine(result.Text);
                    for (int i = 0; i < result.Tokens.Count; i++)
                    {
                        Console.WriteLine(string.Format("{0}:[{1},{2}]", result.Tokens[i], result.Timestamps[i].First(), result.Timestamps[i].Last()));
                    }
                    Console.WriteLine("");
                }
                // Non batch method
            }
            if (streamDecodeMethod == "multi")
            {
                //2. batch method
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
                    for (int i = 0; i < result.Tokens.Count; i++)
                    {
                        Console.WriteLine(string.Format("{0}:[{1},{2}]", result.Tokens[i], result.Timestamps[i].First(), result.Timestamps[i].Last()));
                    }
                    Console.WriteLine("");
                }
            }
            if (_offlineRecognizer != null)
            {
                _offlineRecognizer.Dispose();
                _offlineRecognizer = null;
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("recognition_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration_milliseconds:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("end!");
        }
    }
}

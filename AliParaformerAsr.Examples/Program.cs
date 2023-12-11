using AliParaformerAsr;
using NAudio.Wave;
internal static class Program
{
	[STAThread]
	private static void Main()
	{
        string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
        string modelFilePath = applicationBase + "./"+ modelName + "/model.ts.int8.onnx";
        string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
        string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
        string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
        AliParaformerAsr.OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
        List<float[]>? samples = null;
        TimeSpan total_duration = new TimeSpan(0L);
        if (samples == null)
        {
            samples = new List<float[]>();
            for (int i = 1; i < 2; i++)
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
                sample = sample.Select((float x) => x * 32768f).ToArray();
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
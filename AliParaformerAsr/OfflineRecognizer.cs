// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using AliParaformerAsr.Model;
using AliParaformerAsr.Utils;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

namespace AliParaformerAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private OfflineModel _offlineModel;
        private readonly ILogger<OfflineRecognizer> _logger;
        private string[] _tokens;
        private AsrYamlEntity _asrYamlEntity;
        private string _mvnFilePath;

        public OfflineRecognizer(string modelFilePath, string configFilePath, string mvnFilePath, string tokensFilePath, int batchSize = 1, int threadsNum = 1)
        {
            _offlineModel = new OfflineModel(modelFilePath, threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _asrYamlEntity = YamlHelper.ReadYaml<AsrYamlEntity>(configFilePath);
            _mvnFilePath = mvnFilePath;
            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<OfflineRecognizer>(loggerFactory);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream offlineStream = new OfflineStream(_mvnFilePath, _asrYamlEntity);
            return offlineStream;
        }
        public OfflineRecognizerResultEntity GetResult(OfflineStream stream)
        {
            List<OfflineStream> streams = new List<OfflineStream>();
            streams.Add(stream);
            OfflineRecognizerResultEntity text_result = GetResults(streams)[0];

            return text_result;
        }
        public List<OfflineRecognizerResultEntity> GetResults(List<OfflineStream> streams)
        {
            this._logger.LogInformation("get features begin");
            this.Forward(streams);
            List<OfflineRecognizerResultEntity> text_results = this.DecodeMulti(streams);
            return text_results;
        }

        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            foreach (OfflineStream stream in streams)
            {
                modelInputs.Add(stream.OfflineInputEntity);
            }
            int batchSize = modelInputs.Count;
            float[] padSequence = PadSequence(modelInputs);
            var inputMeta = _offlineModel.ModelSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "speech")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / 560 / batchSize, 560 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "speech_lengths")
                {
                    int[] dim = new int[] { batchSize };
                    int[] speech_lengths = new int[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        speech_lengths[i] = padSequence.Length / 560 / batchSize;
                    }
                    var tensor = new DenseTensor<int>(speech_lengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
            }
            IReadOnlyCollection<string> outputNames = new List<string>();
            outputNames.Append("logits");
            outputNames.Append("token_num");
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;
            try
            {
                results = _offlineModel.ModelSession.Run(container);
                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    Tensor<float> logits_tensor = resultsArray[0].AsTensor<float>();
                    Tensor<Int64> token_nums_tensor = resultsArray[1].AsTensor<Int64>();
                    List<Int64[]> token_nums = new List<Int64[]> { };
                    List<List<int[]>> timestamps_list = new List<List<int[]>>();
                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[logits_tensor.Dimensions[1]];
                        List<int[]> timestamps = new List<int[]>();
                        for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                            {
                                token_num = logits_tensor[i, j, token_num] > logits_tensor[i, j, k] ? token_num : k;
                            }
                            item[j] = (int)token_num;
                            timestamps.Add(new int[] { 0, 0 });
                        }
                        token_nums.Add(item);
                        timestamps_list.Add(timestamps);
                    }
                    if (resultsArray.Length >= 4)
                    {
                        timestamps_list = new List<List<int[]>>();
                        Tensor<float> cif_peak_tensor = resultsArray[3].AsTensor<float>();
                        for (int i = 0; i < cif_peak_tensor.Dimensions[0]; i++)
                        {
                            float[] us_cif_peak = new float[cif_peak_tensor.Dimensions[1]];
                            Array.Copy(cif_peak_tensor.ToArray(), i * us_cif_peak.Length, us_cif_peak, 0, us_cif_peak.Length);
                            List<int[]> timestamps = time_stamp_lfr6_onnx(us_cif_peak);
                            timestamps_list.Add(timestamps);
                        }
                    }
                    int streamIndex = 0;
                    foreach (OfflineStream stream in streams)
                    {
                        stream.Tokens = token_nums[streamIndex].ToList();
                        stream.Timestamps.AddRange(timestamps_list[streamIndex]);
                        stream.RemoveSamples();
                        streamIndex++;
                    }
                }
            }
            catch (Exception ex)
            {
                //
            }
        }

        private List<int[]> time_stamp_lfr6_onnx(float[] us_cif_peak, float begin_time = 0.0F, float total_offset = -1.5F)
        {
            List<float[]> timestamp_list = new List<float[]>();
            int START_END_THRESHOLD = 5;
            int MAX_TOKEN_DURATION = 30;
            float TIME_RATE = 10.0F * 6 / 1000 / 3;//3 times upsampled

            int num_frames = us_cif_peak.Length;
            List<float> fire_place = new List<float>();
            for (int i = 0; i < us_cif_peak.Length; i++)
            {
                if (us_cif_peak[i] > 1.0 - 1e-4)
                {
                    fire_place.Add(i + total_offset);
                }
            }
            //begin silence
            if (fire_place[0] > START_END_THRESHOLD)
            {                
                timestamp_list.Add(new float[] { 0.0f, fire_place[0] * TIME_RATE });
            }
            List<int[]> timestamps = new List<int[]>();
            for (int i = 0; i < fire_place.Count - 1; i++)
            {
                if (i == fire_place.Count - 2 || MAX_TOKEN_DURATION < 0 || fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION)
                {
                    timestamp_list.Add(new float[] { fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE });
                }
                else
                {
                    float _split = fire_place[i] + MAX_TOKEN_DURATION;
                }
            }
            // tail token and end silence
            if (num_frames - fire_place.Last() > START_END_THRESHOLD)
            {
                float _end = (num_frames + fire_place.Last()) / 2;
                timestamp_list.Last()[1] = _end * TIME_RATE;
                timestamp_list.Add(new float[] { _end * TIME_RATE, num_frames * TIME_RATE });
            }
            else
            {
                timestamp_list.Last()[1] = num_frames * TIME_RATE;
            }
            if (begin_time > 0.0F)
            {
                for (int i = 0; i < timestamp_list.Count; i++)
                {
                    timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0F;
                    timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0F;
                }
            }
            foreach (float[] timestamp in timestamp_list)
            {
                timestamps.Add(new int[] { (int)(timestamp[0] * 1000), (int)(timestamp[1] * 1000) });
            }
            return timestamps;
        }

        private List<OfflineRecognizerResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
            List<string> text_results = new List<string>();
#pragma warning disable CS8602 // 解引用可能出现空引用。

            foreach (var stream in streams)
            {
                OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                string text_result = "";
                string lastToken = "";
                int[] lastTimestamp = null;
                foreach (var result in stream.Tokens.Zip<Int64, int[]>(stream.Timestamps))
                {
                    Int64 token = result.First;
                    if (token == 2)
                    {
                        break;
                    }
                    if (_tokens[token] != "</s>" && _tokens[token] != "<s>" && _tokens[token] != "<blank>")
                    {
                        if (IsChinese(_tokens[token], true))
                        {
                            text_result += _tokens[token];
                            offlineRecognizerResultEntity.Tokens.Add(_tokens[token]);
                            offlineRecognizerResultEntity.Timestamps.Add(result.Second);
                        }
                        else
                        {
                            text_result += "▁" + _tokens[token] + "▁";
                            if ((lastToken + "▁" + _tokens[token] + "▁").IndexOf("@@▁▁") > 0)
                            {
                                string currToken = (lastToken + "▁" + _tokens[token] + "▁").Replace("@@▁▁", "");
                                int[] currTimestamp = null;
                                if (lastTimestamp == null)
                                {
                                    currTimestamp = result.Second;
                                }
                                else
                                {
                                    List<int> temp = lastTimestamp.ToList();
                                    temp.AddRange(result.Second.ToList());
                                    currTimestamp = temp.ToArray();
                                }
                                offlineRecognizerResultEntity.Tokens.Remove(offlineRecognizerResultEntity.Tokens.Last());
                                offlineRecognizerResultEntity.Tokens.Add(currToken.Replace("▁", ""));
                                offlineRecognizerResultEntity.Timestamps.Remove(offlineRecognizerResultEntity.Timestamps.Last());
                                offlineRecognizerResultEntity.Timestamps.Add(currTimestamp);
                                lastToken = currToken;
                                lastTimestamp = currTimestamp;
                            }
                            else
                            {
                                offlineRecognizerResultEntity.Tokens.Add(_tokens[token]);
                                offlineRecognizerResultEntity.Timestamps.Add(result.Second);
                                lastToken = "▁" + _tokens[token] + "▁";
                                lastTimestamp = result.Second;
                            }

                        }

                    }
                }
                text_result = text_result.Replace("@@▁▁", "").Replace("▁▁", " ").Replace("@@", " ").Replace("▁", "");
                text_results.Add(text_result);
                offlineRecognizerResultEntity.Text = text_result;
                offlineRecognizerResultEntity.TextLen = text_result.Length;
                offlineRecognizerResultEntities.Add(offlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。
            return offlineRecognizerResultEntities;
        }

        /// <summary>
        /// Verify if the string is in Chinese.
        /// </summary>
        /// <param name="checkedStr">The string to be verified.</param>
        /// <param name="allMatch">Is it an exact match. When the value is true,all are in Chinese; 
        /// When the value is false, only Chinese is included.
        /// </param>
        /// <returns></returns>
        private bool IsChinese(string checkedStr, bool allMatch)
        {
            string pattern;
            if (allMatch)
                pattern = @"^[\u4e00-\u9fa5]+$";
            else
                pattern = @"[\u4e00-\u9fa5]";
            if (Regex.IsMatch(checkedStr, pattern))
                return true;
            else
                return false;
        }

        private float[] PadSequence(List<OfflineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength) + 560 * 19;
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            float[,] xxx = new float[modelInputs.Count, max_speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                if (max_speech_length == modelInputs[i].SpeechLength)
                {
                    for (int j = 0; j < xxx.GetLength(1); j++)
                    {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                        xxx[i, j] = modelInputs[i].Speech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。
                    }
                    continue;
                }
                float[] nullspeech = new float[max_speech_length - modelInputs[i].SpeechLength];
                float[]? curr_speech = modelInputs[i].Speech;
                float[] padspeech = new float[max_speech_length];
                Array.Copy(curr_speech, 0, padspeech, 0, curr_speech.Length);
                for (int j = 0; j < padspeech.Length; j++)
                {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                    xxx[i, j] = padspeech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。 
                }
            }
            int s = 0;
            for (int i = 0; i < xxx.GetLength(0); i++)
            {
                for (int j = 0; j < xxx.GetLength(1); j++)
                {
                    speech[s] = xxx[i, j];
                    s++;
                }
            }
            speech = speech.Select(x => x == 0 ? -23.025850929940457F : x).ToArray();
            return speech;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_offlineModel != null)
                    {
                        _offlineModel.Dispose();
                    }
                    if (_tokens != null)
                    {
                        _tokens = null;
                    }
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~OfflineRecognizer()
        {
            Dispose(_disposed);
        }
    }
}
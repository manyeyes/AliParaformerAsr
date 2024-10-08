﻿// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using AliParaformerAsr.Model;
using AliParaformerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AliParaformerAsr
{
    internal class OfflineProjOfSenseVoiceSmall : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _modelSession;
        private EmbedSVModel _embedModel;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _featureDim = 80;
        private int _sampleRate = 16000;

        private bool _use_itn = false;
        private string _textnorm = "woitn";
        private Dictionary<string, int> _lidDict = new Dictionary<string, int>() { { "auto", 0 }, { "zh", 3 }, { "en", 4 }, { "yue", 7 }, { "ja", 11 }, { "ko", 12 }, { "nospeech", 13 } };
        private Dictionary<int, int> _lidIntDict = new Dictionary<int, int>() { { 24884, 3 }, { 24885, 4 }, { 24888, 7 }, { 24892, 11 }, { 24896, 12 }, { 24992, 13 } };
        private Dictionary<string, int> _textnormDict = new Dictionary<string, int>() { { "withitn", 14 }, { "woitn", 15 } };
        private Dictionary<int, int> _textnormIntDict = new Dictionary<int, int>() { { 25016, 14 }, { 25017, 15 } };

        public OfflineProjOfSenseVoiceSmall(OfflineModel offlineModel)
        {
            _modelSession = offlineModel.ModelSession;
            var inputMeta = _modelSession.InputMetadata;
            if (!inputMeta.ContainsKey("language") && !inputMeta.ContainsKey("textnorm"))
            {
                _embedModel = new EmbedSVModel();
            }
            _blank_id = offlineModel.Blank_id;
            _sos_eos_id = offlineModel.Sos_eos_id;
            _unk_id = offlineModel.Unk_id;
            _featureDim = offlineModel.FeatureDim;
            _sampleRate = offlineModel.SampleRate;
            _use_itn = offlineModel.Use_itn;
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }

        public ModelOutputEntity ModelProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            //
            string languageValue = "ja";
            int languageId = 0;
            if (_lidDict.ContainsKey(languageValue))
            {
                languageId = _lidDict.GetValueOrDefault(languageValue);
            }
            string textnormValue = "withitn";
            if (!_use_itn)
            {
                textnormValue = "woitn";
            }
            int textnormId = 15;
            if (_textnormDict.ContainsKey(textnormValue))
            {
                textnormId = _textnormDict.GetValueOrDefault(textnormValue);
            }
            var inputMeta = _modelSession.InputMetadata;
            if (!inputMeta.ContainsKey("language") && !inputMeta.ContainsKey("textnorm"))
            {
                List<OfflineInputEntity> offlineInputEntities = new List<OfflineInputEntity>();
                foreach (OfflineInputEntity offlineInputEntity in modelInputs)
                {
                    float[]? speech = offlineInputEntity.Speech;
                    if (speech != null)
                    {
                        float[] language_query = _embedModel.Forward(new Int64[] { languageId });
                        float[] textnorm_query = _embedModel.Forward(new long[] { textnormId });
                        //
                        float[] tempSpeech = new float[speech.Length + 560];
                        Array.Copy(textnorm_query, 0, tempSpeech, 0, textnorm_query.Length);
                        Array.Copy(speech, 0, tempSpeech, textnorm_query.Length, speech.Length);
                        speech = tempSpeech;
                        //
                        float[] event_emo_query = _embedModel.Forward(new Int64[] { 1, 2 });
                        float[] input_query = new float[language_query.Length + event_emo_query.Length];
                        Array.Copy(language_query, 0, input_query, 0, language_query.Length);
                        Array.Copy(event_emo_query, 0, input_query, language_query.Length, event_emo_query.Length);
                        //
                        float[] tempSpeech2 = new float[speech.Length + input_query.Length];
                        Array.Copy(input_query, 0, tempSpeech2, 0, input_query.Length);
                        Array.Copy(speech, 0, tempSpeech2, input_query.Length, speech.Length);
                        speech = tempSpeech2;
                    }
                    offlineInputEntity.Speech = speech;
                    offlineInputEntity.SpeechLength = speech.Length;
                    offlineInputEntities.Add(offlineInputEntity);
                }
                modelInputs = offlineInputEntities;
            }
            float[] padSequence = PadHelper.PadSequence(modelInputs);
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
                if (name == "language")
                {
                    int[] language = new int[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        language[i] = languageId;
                    }
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<int>(language, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name == "textnorm")
                {
                    int[] textnorm = new int[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        textnorm[i] = textnormId;
                    }
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<int>(textnorm, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
            }
            ModelOutputEntity modelOutputEntity = new ModelOutputEntity();
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _modelSession.Run(container);

                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    modelOutputEntity.model_out = resultsArray[0].AsTensor<float>();
                    modelOutputEntity.model_out_lens = resultsArray[1].AsEnumerable<int>().ToArray();
                    if (resultsArray.Length >= 4)
                    {
                        Tensor<float> cif_peak_tensor = resultsArray[3].AsTensor<float>();
                        modelOutputEntity.cif_peak_tensor = cif_peak_tensor;
                    }
                }
            }
            catch (Exception ex)
            {
                //
            }
            return modelOutputEntity;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_modelSession != null)
                    {
                        _modelSession.Dispose();
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
        ~OfflineProjOfSenseVoiceSmall()
        {
            Dispose(_disposed);
        }
    }
}

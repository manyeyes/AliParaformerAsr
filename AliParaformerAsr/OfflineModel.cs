// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime;

namespace AliParaformerAsr
{
    public class OfflineModel
    {
        private InferenceSession _modelSession;
        private string _modelebFilePath;
        private List<int[]>? _hotwords = null;
        private int _blank_id = 0;
        private int sos_eos_id = 1;
        private int _unk_id = 2;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private bool _use_itn = false;

        public OfflineModel(string modelFilePath, string modelebFilePath = "", string hotwordFilePath = "", int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
            _modelebFilePath = modelebFilePath;
        }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => sos_eos_id; set => sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public bool Use_itn { get => _use_itn; set => _use_itn = value; }
        public string ModelebFilePath { get => _modelebFilePath; set => _modelebFilePath = value; }
        public List<int[]>? Hotwords { get => _hotwords; set => _hotwords = value; }

        public InferenceSession initModel(string modelFilePath, int threadsNum = 2)
        {
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            //options.AppendExecutionProvider_DML(0);
            options.AppendExecutionProvider_CPU(0);
            //options.AppendExecutionProvider_CUDA(0);
            options.InterOpNumThreads = threadsNum;
            InferenceSession onnxSession = new InferenceSession(modelFilePath, options);
            return onnxSession;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_modelSession != null)
                {
                    _modelSession.Dispose();
                }
            }
        }

        internal void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}

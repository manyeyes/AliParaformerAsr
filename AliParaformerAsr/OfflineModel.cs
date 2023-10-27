// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime;

namespace AliParaformerAsr
{
    public class OfflineModel
    {
        private InferenceSession _modelSession;
        private int _blank_id = 0;
        private int sos_eos_id = 1;
        private int _unk_id = 2;
        private int _featureDim = 80;

        public OfflineModel(string modelFilePath, int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
        }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => sos_eos_id; set => sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }

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

    }
}

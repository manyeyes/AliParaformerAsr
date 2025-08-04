// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime;
using System.Reflection;

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
            _hotwords = GetHotwords(hotwordFilePath);
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
            if (string.IsNullOrEmpty(modelFilePath))
            {
                return null;
            }
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            //options.AppendExecutionProvider_DML(0);
            options.AppendExecutionProvider_CPU(0);
            //options.AppendExecutionProvider_CUDA(0);
            options.InterOpNumThreads = threadsNum;
            InferenceSession onnxSession = null;
            if (!string.IsNullOrEmpty(modelFilePath) && modelFilePath.IndexOf("/") < 0)
            {
                byte[] model = ReadEmbeddedResourceAsBytes(modelFilePath);
                onnxSession = new InferenceSession(model, options);
            }
            else
            {
                onnxSession = new InferenceSession(modelFilePath, options);
            }
            return onnxSession;
        }
        private static byte[] ReadEmbeddedResourceAsBytes(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();
            var stream = assembly.GetManifestResourceStream(resourceName) ??
                         throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");
            byte[] bytes = new byte[stream.Length];
            stream.Read(bytes, 0, bytes.Length);
            stream.Seek(0, SeekOrigin.Begin);
            stream.Close();
            stream.Dispose();
            return bytes;
        }
        private List<int[]>? GetHotwords(string hotwordFilePath = "")
        {
            List<int[]>? hotwords = null;
            //TODO: read data from hotwordFilePath
            return hotwords;
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

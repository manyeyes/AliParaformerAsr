// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace AliParaformerAsr
{
    public class EmbedModel
    {
        private InferenceSession _modelSession;

        public EmbedModel(int threadsNum = 2)
        {
            _modelSession = initModel(threadsNum);
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }

        public InferenceSession initModel(int threadsNum = 2)
        {
            byte[] model = ReadEmbeddedResourceAsBytes("AliParaformerAsr.data.embed.onnx");
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            //options.AppendExecutionProvider_DML(0);
            options.AppendExecutionProvider_CPU(0);
            //options.AppendExecutionProvider_CUDA(0);
            options.InterOpNumThreads = threadsNum;
            InferenceSession onnxSession = new InferenceSession(model, options);
            return onnxSession;
        }
        private static byte[] ReadEmbeddedResourceAsBytes(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();

            var stream = assembly.GetManifestResourceStream(resourceName) ??
                         throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");

            byte[] bytes = new byte[stream.Length];
            stream.Read(bytes, 0, bytes.Length);

            // 设置当前流的位置为流的开始 
            stream.Seek(0, SeekOrigin.Begin);
            //var reader = new StreamReader(stream);
            //var content = reader.ReadToEnd();
            //byte[] bytes = stream.ReadByte();// Encoding.UTF8.GetBytes(reader.ReadToEnd());

            //reader.Close();
            //reader.Dispose();
            stream.Close();
            stream.Dispose();

            return bytes;
        }
        public float[] Forward(Int64[] x,int speechSize=0)
        {
            float[] y=new float[0];
            var inputMeta = _modelSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int[] dim = new int[] { 1,x.Length };
                    var tensor = new DenseTensor<Int64>(x, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
            }
            IReadOnlyCollection<string> outputNames = new List<string>();
            outputNames.Append("y");
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;
            try
            {
                results = _modelSession.Run(container);
                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    Tensor<float> logits_tensor = resultsArray[0].AsTensor<float>();
                    y = logits_tensor.ToArray();
                }
            }
            catch (Exception ex)
            {
                //
            }
            return y;
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

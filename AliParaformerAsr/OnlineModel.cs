// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace AliParaformerAsr
{
    public class OnlineModel
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;

        private int _lfr = 10;
        private int _chunkLength;
        private int _shiftLength;
        
        public OnlineModel(string encoderFilePath, string decoderFilePath, int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _shiftLength = _chunkLength;
            _chunkLength = _lfr * 6;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }

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
        public List<float[]> StackCifHiddens(List<List<float[]>> cifHiddens)
        {
            int batchSize = cifHiddens.Count;
            List<float[]> cifHidden = new List<float[]>();
            for (int b = 0; b < batchSize; b++)
            {
                foreach (float[] item in cifHiddens[b])
                {
                    cifHidden.Add(item);
                }
            }
            return cifHidden;
        }

        public List<List<float[]>> UnStackCifHiddens(List<float[]> cifHidden, int batchSize)
        {
            List<List<float[]>> cifHiddens = new List<List<float[]>>();
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> hiddensItem = new List<float[]>();
                for(int x= b*(cifHidden.Count / batchSize); x < (b+1) * (cifHidden.Count / batchSize); x++)
                {
                    hiddensItem.Add(cifHidden[x]);
                }
                cifHiddens.Add(hiddensItem);
            }
            return cifHiddens;
        }

        public List<List<float>> DynamicMask(List<List<float>> alphas)
        {
            List<List<float>> newAlphas = new List<List<float>>();
            foreach (List<float> item in alphas)
            {
                float[] cifAlphasItem = item.ToArray();
                float[] chunk_size_5 = new float[5];
                if (cifAlphasItem.Length > 5)
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, 5);
                }
                else
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, cifAlphasItem.Length);
                }
                int decodeLfr = 5 + _lfr;
                if (cifAlphasItem.Length > decodeLfr)
                {
                    float[] chunk_size_15 = new float[cifAlphasItem.Length - decodeLfr];
                    Array.Copy(chunk_size_15, 0, cifAlphasItem, decodeLfr, chunk_size_15.Length);
                }
                newAlphas.Add(cifAlphasItem.ToList());
            }
            return newAlphas;
        }

        public float[] StackCifAlphas(List<float[]> cifAlphas)
        {
            int batchSize = cifAlphas.Count;
            float[] cifAlpha = new float[cifAlphas[0].Length * batchSize];
            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(cifAlphas[b], 0, cifAlpha, b * cifAlphas[0].Length, cifAlphas[b].Length);
            }
            return cifAlpha;
        }

        public List<float[]> UnStackCifAlphas(float[] cifAlpha, int batchSize)
        {
            List<float[]> cifAlphas = new List<float[]>();
            for (int b = 0; b < batchSize; b++)
            {
                float[] cifAlphasItem = new float[cifAlpha.Length / batchSize];
                Array.Copy(cifAlpha, b * cifAlphasItem.Length, cifAlphasItem, 0, cifAlphasItem.Length);
                //////////
                float[] chunk_size_5 = new float[5];                
                if (cifAlphasItem.Length > 5)
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, 5);
                }
                else
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, cifAlphasItem.Length);
                }
                if (cifAlphasItem.Length > 15)
                {
                    float[] chunk_size_15 = new float[cifAlphasItem.Length-15];
                    Array.Copy(chunk_size_15, 0, cifAlphasItem, 15, chunk_size_15.Length);
                }
                //////////
                cifAlphas.Add(cifAlphasItem);
            }
            return cifAlphas;
        }

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            int batchSize = statesList.Count;
            Debug.Assert(statesList[0].Count % 16 == 0, "when stack_states, state_list[0] is 16x");
            int fsmnLayer = statesList[0].Count;
            for (int i = 0; i < fsmnLayer; i++)
            {
                float[] statesItemTemp = new float[statesList[0][i].Length * batchSize];
                int statesItemTemp_item_length = statesList[0][i].Length;
                int statesItemTemp_item_axisnum = 512 * 10;
                for (int x = 0; x < statesItemTemp_item_length / statesItemTemp_item_axisnum; x++)
                {
                    for (int n = 0; n < batchSize; n++)
                    {
                        float[] statesItemTemp_item = statesList[n][0];
                        Array.Copy(statesItemTemp_item, x * statesItemTemp_item_axisnum, statesItemTemp, (x * batchSize + n) * statesItemTemp_item_axisnum, statesItemTemp_item_axisnum);
                    }
                }
                states.Add(statesItemTemp);
            }
            return states;
        }
        public List<List<float[]>> unstack_states(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 16 == 0, "when stack_states, state_list[0] is 16x");
            int fsmnLayer = states.Count;
            int batchSize = states[0].Length / 512 / 10;
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> statesListItem = new List<float[]>();
                for (int j = 0; j < fsmnLayer; j++)
                {
                    float[] item = states[j];
                    int statesItemTemp_axisnum = 512 * 10;
                    int statesItemTemp_size = 1 * 512 * 10;
                    float[] statesItemTemp_item = new float[statesItemTemp_size];
                    for (int k = 0; k < statesItemTemp_size / statesItemTemp_axisnum; k++)
                    {
                        Array.Copy(item, (item.Length / statesItemTemp_size * k + b) * statesItemTemp_axisnum, statesItemTemp_item, k * statesItemTemp_axisnum, statesItemTemp_axisnum);
                    }
                    statesListItem.Add(statesItemTemp_item);
                }
                statesList.Add(statesListItem);
            }
            return statesList;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_encoderSession != null)
                {
                    _encoderSession.Dispose();
                }
                if (_decoderSession != null)
                {
                    _decoderSession.Dispose();
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

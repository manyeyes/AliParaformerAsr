// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Reflection;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AliParaformerAsr
{
    public class OnlineStream
    {
        private FrontendConfEntity _frontendConfEntity;
        private int _fsmnDims;
        private int _fsmnLorder;
        private int _fsmnLayer;
        private OnlineWavFrontend _wavFrontend;
        private OnlineInputEntity _onlineInputEntity;
        private int _blank_id = 0;
        private int _unk_id = 2;
        private Int64[] _hyp;
        private int _chunkLength;

        private List<Int64> _tokens = new List<Int64>();
        private List<int> _timestamps = new List<int>();
        private List<float[]> _states = new List<float[]>();
        private List<float[]> _cifHidden = new List<float[]>();
        private List<float> _cifAlpha = new List<float>();
        private int _startIdx = 0;
        private static object obj = new object();
        private float[] _cacheFeats = null;
        private float[] _cacheInput = null;
        private float[] _cachelfrSplice = null;
        private int _frame_sample_length;
        private int _frame_shift_sample_length;
        private int _lfr_m = 7;
        private float[] _cacheSamples = null;
        public OnlineStream(string mvnFilePath, AsrYamlEntity asrYamlEntity, int chunkLength)
        {
            _onlineInputEntity = new OnlineInputEntity();
            _frontendConfEntity = asrYamlEntity.frontend_conf;
            _fsmnDims = asrYamlEntity.encoder_conf.output_size;
            _fsmnLorder = asrYamlEntity.decoder_conf.kernel_size - 1;
            _fsmnLayer = asrYamlEntity.decoder_conf.num_blocks;
            _wavFrontend = new OnlineWavFrontend(mvnFilePath, asrYamlEntity.frontend_conf);
            _hyp = new Int64[] { _blank_id, _blank_id };
            _states = InitEncoderStates();
            _cifHidden = InitHidden();
            _cifAlpha = InitAlpha();
            _cacheFeats = InitCacheFeats();
            _cacheSamples = new float[160* chunkLength];
            _chunkLength = chunkLength;
            _tokens = new List<Int64> { _blank_id, _blank_id };
            _frame_sample_length = 25 * 16000 / 1000;
            _frame_shift_sample_length = 10 * 16000 / 1000;
        }

        public OnlineInputEntity OnlineInputEntity { get => _onlineInputEntity; set => _onlineInputEntity = value; }
        public long[] Hyp { get => _hyp; set => _hyp = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public List<float[]> CifHidden { get => _cifHidden; set => _cifHidden = value; }
        public List<float> CifAlpha { get => _cifAlpha; set => _cifAlpha = value; }

        private int ComputeFrameNum(int samplesLength)
        {
            int frameNum = (samplesLength - _frame_sample_length) / _frame_shift_sample_length + 1;
            if (frameNum < 1 || samplesLength < _frame_sample_length)
            {
                frameNum = 0;
            }
            return frameNum;
        }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                if (_cacheSamples.Length > 0)
                {
                    oLen = _cacheSamples.Length;
                }
                float[]? samplesTemp = new float[oLen + samples.Length];
                if (oLen > 0)
                {
                    Array.Copy(_cacheSamples, 0, samplesTemp, 0, oLen);
                }
                Array.Copy(samples, 0, samplesTemp, oLen, samples.Length);
                _cacheSamples = samplesTemp;
                int cacheSamplesLength = _cacheSamples.Length;
                int chunkSamplesLength = 160 * _chunkLength;
                if (cacheSamplesLength > chunkSamplesLength)
                {
                    float[] _samples = new float[chunkSamplesLength];
                    Array.Copy(_cacheSamples, 0, _samples, 0, _samples.Length);
                    InputSpeech(_samples);
                    float[] _cacheSamplesTemp = new float[cacheSamplesLength - chunkSamplesLength];
                    Array.Copy(_cacheSamples, chunkSamplesLength, _cacheSamplesTemp, 0, _cacheSamplesTemp.Length);
                    _cacheSamples = _cacheSamplesTemp;
                }
            }
        }

        public void InputSpeech(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    oLen = OnlineInputEntity.SpeechLength;
                }
                float[] inputs = new float[samples.Length];
                if (_cacheInput != null)
                {
                    inputs = new float[_cacheInput.Length + samples.Length];
                    Array.Copy(_cacheInput, 0, inputs, 0, _cacheInput.Length);
                    Array.Copy(samples, 0, inputs, _cacheInput.Length, samples.Length);
                }
                else
                {
                    Array.Copy(samples, 0, inputs, 0, samples.Length);
                }
                int frameNum = ComputeFrameNum(inputs.Length);
                int waveformLength = inputs.Length;
                float[] waveform = new float[waveformLength];
                Array.Copy(inputs, 0, waveform, 0, waveform.Length);
                waveform = waveform.Select((float x) => x * 32768f).ToArray();
                float[] features = _wavFrontend.GetFbank(waveform);
                if (_cacheInput == null)
                {
                    int repeatNum = (_lfr_m - 1) / 2 - 1;
                    int featureDim = _frontendConfEntity.n_mels;
                    float[] firstFbank = new float[featureDim];
                    Array.Copy(features, 0, firstFbank, 0, firstFbank.Length);
                    float[] features_temp = new float[featureDim * repeatNum + features.Length];
                    for (int i = 0; i < repeatNum; i++)
                    {
                        Array.Copy(firstFbank, 0, features_temp, i * featureDim, featureDim);
                    }
                    Array.Copy(features, 0, features_temp, featureDim * repeatNum, features.Length);
                    features = features_temp;
                }
                // compute cacheInput
                int cacheInputLength = inputs.Length - frameNum * _frame_shift_sample_length;
                _cacheInput = new float[cacheInputLength];
                Array.Copy(inputs, inputs.Length - cacheInputLength, _cacheInput, 0, cacheInputLength);
                float[]? featuresTemp = new float[oLen + features.Length];
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    Array.Copy(_onlineInputEntity.Speech, 0, featuresTemp, 0, _onlineInputEntity.SpeechLength);
                }
                Array.Copy(features, 0, featuresTemp, OnlineInputEntity.SpeechLength, features.Length);
                OnlineInputEntity.Speech = featuresTemp;
                OnlineInputEntity.SpeechLength = featuresTemp.Length;
            }
        }

        // Note: chunk_length is in frames before subsampling
        public float[]? GetDecodeChunk(int chunkLength)
        {
            //chunkLength = _chunkLength;
            int featureDim = _frontendConfEntity.n_mels;
            lock (obj)
            {
                float[]? decodeChunk = null;
                if (chunkLength * featureDim <= _onlineInputEntity.SpeechLength)
                {

                    float[] padChunk = new float[chunkLength * featureDim];
                    float[]? features = _onlineInputEntity.Speech;
                    Array.Copy(features, 0, padChunk, 0, padChunk.Length);
                    if (_cachelfrSplice != null)
                    {
                        float[] padChunk_temp = new float[chunkLength * featureDim + _cachelfrSplice.Length];
                        Array.Copy(_cachelfrSplice, 0, padChunk_temp, 0, _cachelfrSplice.Length);
                        Array.Copy(padChunk, 0, padChunk_temp, _cachelfrSplice.Length, padChunk.Length);
                        padChunk = padChunk_temp;
                    }
                    else
                    {
                        float[] firstFbank = new float[80];
                        Array.Copy(padChunk, 0, firstFbank, 0, firstFbank.Length);
                        float[] padChunk_temp = new float[chunkLength * featureDim + firstFbank.Length];
                        Array.Copy(firstFbank, 0, padChunk_temp, 0, firstFbank.Length);
                        Array.Copy(padChunk, 0, padChunk_temp, firstFbank.Length, padChunk.Length);
                        padChunk = padChunk_temp;
                    }
                    //_cachelfrSplice
                    _cachelfrSplice = new float[featureDim];
                    Array.Copy(padChunk, padChunk.Length - _cachelfrSplice.Length, _cachelfrSplice, 0, _cachelfrSplice.Length);
                    //缓存
                    padChunk = _wavFrontend.LfrCmvn(padChunk);
                    padChunk = padChunk.Select(x => (float)(x * Math.Pow(512, 0.5))).ToArray();
                    //position encoding
                    int timesteps = padChunk.Length / 560;
                    padChunk = _wavFrontend.SinusoidalPositionEncoder(padChunk, timesteps, 560, _startIdx);
                    decodeChunk = new float[_cacheFeats.Length + padChunk.Length];
                    Array.Copy(_cacheFeats, 0, decodeChunk, 0, _cacheFeats.Length);
                    Array.Copy(padChunk, 0, decodeChunk, _cacheFeats.Length, padChunk.Length);
                    _startIdx += timesteps;
                    Array.Copy(decodeChunk, decodeChunk.Length - _cacheFeats.Length, _cacheFeats, 0, _cacheFeats.Length);
                    RemoveChunk(chunkLength);
                }
                return decodeChunk;
            }
        }

        public void RemoveChunk(int shiftLength)
        {
            lock (obj)
            {
                int featureDim = _frontendConfEntity.n_mels;
                if (shiftLength * featureDim <= _onlineInputEntity.SpeechLength)
                {
                    float[]? features = _onlineInputEntity.Speech;
                    float[]? featuresTemp = new float[features.Length - shiftLength * featureDim];
                    Array.Copy(features, shiftLength * featureDim, featuresTemp, 0, featuresTemp.Length);
                    _onlineInputEntity.Speech = featuresTemp;
                    _onlineInputEntity.SpeechLength = featuresTemp.Length;
                }
            }
        }

        private List<float[]> InitHidden(int batchSize = 1)
        {
            List<float[]> hidden = new List<float[]>();
            for (int i = 0; i < batchSize; i++)
            {
                float[] item = new float[1 * _fsmnDims];
                hidden.Add(item);
            }
            return hidden;
        }

        private List<float> InitAlpha(int batchSize = 1)
        {
            List<float> alpha = new List<float>();
            for (int i = 0; i < batchSize; i++)
            {
                float item = 0F;
                alpha.Add(item);
            }
            return alpha;
        }


        private List<float[]> InitEncoderStates(int batchSize = 1)
        {
            List<float[]> states = new List<float[]>();
            for (int i = 0; i < _fsmnLayer; i++)
            {
                int fsmn_cache_size = batchSize * _fsmnDims * _fsmnLorder;
                float[] fsmn_cache = new float[fsmn_cache_size];
                states.Add(fsmn_cache);
            }
            return states;
        }

        private float[] InitCacheFeats(int batchSize = 1)
        {
            float[] cacheFeats = new float[batchSize * 10 * 560];
            return cacheFeats;
        }
    }
}

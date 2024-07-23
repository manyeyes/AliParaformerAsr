using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using AliParaformerAsr.Model;
//using HHFramework;
using MathNet.Numerics.IntegralTransforms;

namespace AliParaformerAsr
{
    public class WavFrontend
    {
        private FrontendConfEntity mFrontendConfEntity;
        private CmvnEntity mCmvnEntity;
        private const int mNumFilt = 80;
        private int mSampleRate;
        private int mFrameLenSamples;
        private float[] mWindowFunc;
        private int mFrameShiftSamples;

        public WavFrontend(string mvnFilePath, FrontendConfEntity frontendConfEntity)
        {
            mFrontendConfEntity = frontendConfEntity;
            mSampleRate = frontendConfEntity.fs;
            mCmvnEntity = LoadCmvn(mvnFilePath);
        }

        private void LogFBank(ref float[][] fBank)
        {
            for (int i = 0; i < fBank.Length; ++i)
            {
                for (int j = 0; j < fBank[0].Length; ++j)
                {
                    fBank[i][j] = (float)Math.Log(fBank[i][j]);
                }
            }
        }

        private float[] ConvertFBank1D(float[][] fBank)
        {
            var cols = fBank.Length;
            var rows = fBank[0].Length;
            var fBank1D = new float[cols * rows];
            var index = 0;
            for (var i = 0; i < cols; ++i)
            {
                for (var j = 0; j < rows; ++j)
                {
                    fBank1D[index++] = fBank[i][j];
                }
            }

            return fBank1D;
        }

        public float[] GetFbank(float[] wavData)
        {
            // 将数据归一化到 -1 到 1 之间
            // 音频数据是16位的PCM编码，它的取值范围是从 -32768 到 32767
            wavData = wavData.Select(x => x * 32768f).ToArray();
            var fBank = ExtractFBankFeatures(wavData, mSampleRate, mNumFilt, 400);
            // 取log
            LogFBank(ref fBank);

            var fBank1D = ConvertFBank1D(fBank);
            return fBank1D;
        }

        private float[][] ExtractFBankFeatures(float[] signal, int sampleRate, int numFilters, int fftSize)
        {
            // 预加重
            PreEmphasis(ref signal);
            // 帧划分与加窗
            mFrameLenSamples = 25 * mSampleRate / 1000;
            mWindowFunc = HammingWindow(mFrameLenSamples);
            mFrameShiftSamples = 10 * mSampleRate / 1000;

            var frames = FrameSignal(signal, mFrameLenSamples, mFrameShiftSamples);
            ApplyWindow(ref frames, mWindowFunc);
            // 计算功率谱
            var powerSpectrum = ComputePowerSpectrum(frames, fftSize);
            // 计算Mel滤波器组和FBank特征
            var fBank = MelFilterBank(powerSpectrum, sampleRate, numFilters, fftSize);

            return fBank;
        }

        private void PreEmphasis(ref float[] signal, float preEmphasis = 0.97f)
        {
            for (int i = signal.Length - 1; i > 0; i--)
            {
                signal[i] -= preEmphasis * signal[i - 1];
            }
        }

        private float[][] FrameSignal(float[] signal, int frameLen, int frameShift)
        {
            int numFrames = (int)Math.Ceiling((double)(signal.Length - frameLen) / frameShift) + 1;
            var frames = new float[numFrames][];

            for (int i = 0; i < numFrames; i++)
            {
                frames[i] = new float[frameLen];
                for (int j = 0; j < frameLen; j++)
                {
                    int index = i * frameShift + j;
                    frames[i][j] = index < signal.Length ? signal[index] : 0;
                }
            }

            return frames;
        }

        private void ApplyWindow(ref float[][] frames, float[] window)
        {
            for (int i = 0; i < frames.Length; i++)
            {
                for (int j = 0; j < frames[i].Length; j++)
                {
                    frames[i][j] *= window[j];
                }
            }
        }

        private float[][] MelFilterBank(float[][] powerSpectrum, int sampleRate, int numFilters, int fftSize)
        {
            int numFrames = powerSpectrum.Length;
            float[][] fbank = new float[numFrames][];
            double[] melPoints = new double[numFilters + 2];
            double[] hzPoints = new double[numFilters + 2];
            int[] bin = new int[numFilters + 2];

            // Compute Mel filter bank parameters
            double lowFreq = 0;
            double highFreq = sampleRate >> 1;
            double lowMel = 2595 * Math.Log10(1 + lowFreq / 700);
            double highMel = 2595 * Math.Log10(1 + highFreq / 700);
            double melStep = (highMel - lowMel) / (numFilters + 1);

            for (int i = 0; i < numFilters + 2; i++)
            {
                melPoints[i] = lowMel + i * melStep;
                hzPoints[i] = 700 * (Math.Pow(10, melPoints[i] / 2595) - 1);
                bin[i] = (int)Math.Floor((fftSize + 1) * hzPoints[i] / sampleRate);
            }

            // Apply Mel filter bank
            for (int i = 0; i < numFrames; i++)
            {
                fbank[i] = new float[numFilters];
                for (int j = 0; j < numFilters; j++)
                {
                    double sum = 0;
                    for (int k = bin[j]; k < bin[j + 1]; k++)
                    {
                        sum += (k - bin[j]) / (double)(bin[j + 1] - bin[j]) * powerSpectrum[i][k];
                    }

                    for (int k = bin[j + 1]; k < bin[j + 2]; k++)
                    {
                        sum += (bin[j + 2] - k) / (double)(bin[j + 2] - bin[j + 1]) * powerSpectrum[i][k];
                    }

                    fbank[i][j] = (float)Math.Max(sum, 1e-10); // Apply log and avoid log(0)
                }
            }

            return fbank;
        }

        private float[] HammingWindow(int frameLen)
        {
            float[] window = new float[frameLen];
            for (int i = 0; i < frameLen; i++)
            {
                window[i] = 0.54f - 0.46f * (float)Math.Cos(2 * Math.PI * i / (frameLen - 1));
            }

            return window;
        }

        private float[][] ComputePowerSpectrum(float[][] frames, int fftSize)
        {
            int numFrames = frames.Length;
            float[][] powerSpectrum = new float[numFrames][];

            for (int i = 0; i < numFrames; i++)
            {
                var fftResult = new Complex[fftSize];
                FFT(frames[i], fftSize, ref fftResult); // Implement FFT function separately

                powerSpectrum[i] = new float[fftSize / 2 + 1];
                for (int j = 0; j < fftSize / 2 + 1; j++)
                {
                    powerSpectrum[i][j] = (float)((fftResult[j].Magnitude * fftResult[j].Magnitude) / fftSize);
                }
            }

            return powerSpectrum;
        }

        private void FFT(float[] data, int fftSize, ref Complex[] fftResult)
        {
            Complex[] fftBuffer = new Complex[fftSize];

            for (int i = 0; i < data.Length; i++)
            {
                fftBuffer[i] = new Complex(data[i], 0);
            }

            // Zero padding if data length is less than fftSize
            for (int i = data.Length; i < fftSize; i++)
            {
                fftBuffer[i] = Complex.Zero;
            }

            Fourier.Forward(fftBuffer, FourierOptions.Matlab);

            for (int i = 0; i < fftSize; i++)
            {
                fftResult[i] = fftBuffer[i];
            }
        }

        public float[] LfrCmvn(float[] fBanks)
        {
            var features = fBanks;
            if (mFrontendConfEntity.lfr_m != 1 || mFrontendConfEntity.lfr_n != 1)
            {
                features = ApplyLfr(fBanks, mFrontendConfEntity.lfr_m, mFrontendConfEntity.lfr_n);
            }

            if (mCmvnEntity != null)
            {
                features = ApplyCmvn(features);
            }

            return features;
        }

        public float[] ApplyCmvn(float[] inputs)
        {
            var arrNegMean = mCmvnEntity.Means;
            var negMean = arrNegMean.Select(x => (float)Convert.ToDouble(x)).ToArray();
            var arrInvStddev = mCmvnEntity.Vars;
            var invStddev = arrInvStddev.Select(x => (float)Convert.ToDouble(x)).ToArray();

            var dim = negMean.Length;
            var numFrames = inputs.Length / dim;

            for (int i = 0; i < numFrames; ++i)
            {
                for (int k = 0; k != dim; ++k)
                {
                    inputs[dim * i + k] = (inputs[dim * i + k] + negMean[k]) * invStddev[k];
                }
            }

            return inputs;
        }

        public float[] ApplyLfr(float[] inputs, int lfr_m, int lfr_n)
        {
            var t = inputs.Length / mNumFilt;
            var tLfr = (int)Math.Floor((double)(t / lfr_n));
            var input0 = new float[mNumFilt];
            Array.Copy(inputs, 0, input0, 0, mNumFilt);
            var tileX = (lfr_m - 1) / 2;
            t += tileX;
            var inputsTemp = new float[t * mNumFilt];
            for (var i = 0; i < tileX; ++i)
            {
                Array.Copy(input0, 0, inputsTemp, tileX * mNumFilt, mNumFilt);
            }

            Array.Copy(inputs, 0, inputsTemp, tileX * mNumFilt, inputs.Length);
            inputs = inputsTemp;

            var lfrOutputs = new float[tLfr * lfr_m * mNumFilt];
            for (int i = 0; i < tLfr; ++i)
            {
                if (lfr_m <= t - i * lfr_n)
                {
                    Array.Copy(inputs, i * lfr_n * mNumFilt, lfrOutputs, i * lfr_m * mNumFilt, lfr_m * mNumFilt);
                }
                else
                {
                    // process last LFR frame
                    var numPadding = lfr_m - (t - i * lfr_n);
                    var frame = new float[lfr_m * mNumFilt];
                    Array.Copy(inputs, i * lfr_n * mNumFilt, frame, 0, (t - i * lfr_n) * mNumFilt);

                    for (var j = 0; j < numPadding; ++j)
                    {
                        Array.Copy(inputs, (t - 1) * mNumFilt, frame, (lfr_m - numPadding + j) * mNumFilt, mNumFilt);
                    }

                    Array.Copy(frame, 0, lfrOutputs, i * lfr_m * mNumFilt, frame.Length);
                }
            }

            return lfrOutputs;
        }

        private CmvnEntity LoadCmvn(string mvnFilePath)
        {
            var meansList = new List<float>();
            var varsList = new List<float>();
            var srtReader = new StreamReader(mvnFilePath);
            var i = 0;
            while (!srtReader.EndOfStream)
            {
                var strLine = srtReader.ReadLine();
                if (string.IsNullOrEmpty(strLine)) continue;
                if (strLine.StartsWith("<AddShift>"))
                {
                    i = 1;
                    continue;
                }

                if (strLine.StartsWith("<Rescale>"))
                {
                    i = 2;
                    continue;
                }

                if (strLine.StartsWith("<LearnRateCoef>") && i == 1)
                {
                    var addShiftLine = strLine.Substring(strLine.IndexOf("[", StringComparison.Ordinal) + 1,
                        strLine.LastIndexOf("]", StringComparison.Ordinal) - strLine.IndexOf("[", StringComparison.Ordinal) - 1).Split(" ");
                    meansList = addShiftLine.Where(x => !string.IsNullOrEmpty(x))
                        .Select(x => float.Parse(x.Trim())).ToList();
                    //i++;
                    continue;
                }

                if (!strLine.StartsWith("<LearnRateCoef>") || i != 2) continue;
                {
                    var rescaleLine = strLine.Substring(strLine.IndexOf("[", StringComparison.Ordinal) + 1,
                        strLine.LastIndexOf("]", StringComparison.Ordinal) - strLine.IndexOf("[", StringComparison.Ordinal) - 1).Split(" ");
                    varsList = rescaleLine.Where(x => !string.IsNullOrEmpty(x)).Select(x => float.Parse(x.Trim()))
                        .ToList();
                    //i++;
                }
            }

            var cmvnEntity = new CmvnEntity
            {
                Means = meansList,
                Vars = varsList
            };
            return cmvnEntity;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposing) return;
            mCmvnEntity = null;
            mFrontendConfEntity = null;
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
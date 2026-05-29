# AliParaformerAsr

## Introduction

**AliParaformerAsr** is a speech recognition library written in C#. It uses `Microsoft.ML.OnnxRuntime` as the backend to decode ONNX models. It supports .NET Framework 4.6.1+, .NET 6.0+, .NET Core 3.1, and .NET Standard 2.0+, runs cross-platform, and supports AOT compilation. Easy to use and integrate.

## How to Run the Sample Projects

### 1. Clone the repository

```bash
cd /path/to
git clone https://github.com/manyeyes/AliParaformerAsr.git
```

### 2. Download models (optional – samples will auto-download)

```bash
cd /path/to/AliParaformerAsr/AliParaformerAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[model_name].git
```

### 3. Load the project

Use Visual Studio 2022 (or any other IDE that supports .NET).

### 4. Run a sample project

- **AliParaformerAsr.Examples**: Console / desktop sample demonstrating basic features (offline transcription, real‑time recognition).
- **MauiApp1**: Cross‑platform .NET MAUI sample that runs on Android, iOS, Windows, etc.

### 5. Configuration (`asr.yaml`)

Most parameters in `asr.yaml` can stay as they are. Adjustable parameters:

| Parameter | Description |
| --------- | ----------- |
| `use_itn` | Set to `true` for the SenseVoiceSmall model to enable inverse text normalization. |

## How to Use in Your Code

### Offline (Non‑streaming) Model

#### 1. Add using directive

```csharp
using AliParaformerAsr;
```

#### 2. Model initialization

**Standard Paraformer model:**

```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
string modelFilePath = Path.Combine(applicationBase, modelName, "model_quant.onnx");
string configFilePath = Path.Combine(applicationBase, modelName, "asr.yaml");
string mvnFilePath = Path.Combine(applicationBase, modelName, "am.mvn");
string tokensFilePath = Path.Combine(applicationBase, modelName, "tokens.txt");

OfflineRecognizer offlineRecognizer = new OfflineRecognizer(
    modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
```

**SeACo‑Paraformer model (supports hotwords):**

1. Edit `hotword.txt` in the model directory, one Chinese word per line.
2. Additional parameters are required.

```csharp
string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline";
string modelFilePath = Path.Combine(applicationBase, modelName, "model.int8.onnx");
string modelebFilePath = Path.Combine(applicationBase, modelName, "model_eb.int8.onnx");
string configFilePath = Path.Combine(applicationBase, modelName, "asr.yaml");
string mvnFilePath = Path.Combine(applicationBase, modelName, "am.mvn");
string hotwordFilePath = Path.Combine(applicationBase, modelName, "hotword.txt");
string tokensFilePath = Path.Combine(applicationBase, modelName, "tokens.txt");

OfflineRecognizer offlineRecognizer = new OfflineRecognizer(
    modelFilePath: modelFilePath,
    configFilePath: configFilePath,
    mvnFilePath: mvnFilePath,
    tokensFilePath: tokensFilePath,
    modelebFilePath: modelebFilePath,
    hotwordFilePath: hotwordFilePath);
```

#### 3. Recognition

```csharp
List<float[]> samples = new List<float[]>();
// (Code to read samples from wav file is omitted; see the example project)

List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}

List<Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```

#### 4. Example output

```
欢迎大家来体验达摩院推出的语音识别模型
非常的方便但是现在不同啊英国脱欧欧盟内部完善的产业链的红利人
he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉
elapsed_milliseconds:1502.8828125
total_duration:40525.6875
rtf:0.037084696280599808
end!
```

### Real‑time (Streaming) Model

#### 1. Add using directive

```csharp
using AliParaformerAsr;
```

#### 2. Model initialization

```csharp
string encoderFilePath = Path.Combine(applicationBase, modelName, "encoder.int8.onnx");
string decoderFilePath = Path.Combine(applicationBase, modelName, "decoder.int8.onnx");
string configFilePath = Path.Combine(applicationBase, modelName, "asr.yaml");
string mvnFilePath = Path.Combine(applicationBase, modelName, "am.mvn");
string tokensFilePath = Path.Combine(applicationBase, modelName, "tokens.txt");

OnlineRecognizer onlineRecognizer = new OnlineRecognizer(
    encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
```

#### 3. Recognition

**Batch processing:**

```csharp
List<float[]> samples = new List<float[]>();
// Read samples...

List<OnlineStream> streams = new List<OnlineStream>();
foreach (var sample in samples)
{
    OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OnlineRecognizerResultEntity> results = onlineRecognizer.GetResults(streams);
```

**Single stream:**

```csharp
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
stream.AddSamples(sample);
OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
```

#### 4. Example output

```
正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界
elapsed_milliseconds:1389.3125
total_duration:13052
rtf:0.10644441464909593
Hello, World!
```

## Related Projects

| Project | Description | Repository |
| ------- | ----------- | ---------- |
| AliFsmnVad | Voice activity detection (VAD) for proper segmentation of long audio | [GitHub](https://github.com/manyeyes/AliFsmnVad) |
| AliCTTransformerPunc | Punctuation prediction for recognition results | [GitHub](https://github.com/manyeyes/AliCTTransformerPunc) |

## Additional Information

- **Test project**: `AliParaformerAsr.Examples`
- **Test CPU**: Intel(R) Core(TM) i7-10750H @ 2.60GHz
- **Supported platforms**:
  - Windows 7 SP1 or higher
  - macOS 10.13 (High Sierra) or higher (including iOS)
  - Linux distributions (see .NET 6 supported distributions)
  - Android 5.0 (API 21) or higher

## Model Download (ONNX)

| Model | Type | Languages | Punctuation | Timestamps | Download Links |
| ----- | ---- | --------- | ----------- | ---------- | --------------- |
| paraformer-large-zh-en-onnx-offline | Offline | Chinese, English | No | No | [🤗 HuggingFace](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx) · [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-offline) |
| paraformer-large-zh-en-timestamp-onnx-offline | Offline | Chinese, English | No | Yes | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-timestamp-onnx-offline) |
| paraformer-large-en-onnx-offline | Offline | English | No | No | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-en-onnx-offline) |
| paraformer-large-zh-en-onnx-online | Streaming | Chinese, English | No | No | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-online) |
| paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 | Offline | Chinese, Cantonese, English | No | Yes | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805) |
| paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 | Offline | Chinese, Cantonese, English | No | No | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805) |
| paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 | Streaming | Chinese, Cantonese, English | No | No | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208) |
| paraformer-seaco-large-zh-timestamp-onnx-offline | Offline | Chinese (hotwords) | No | Yes | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline) |
| SenseVoiceSmall | Offline | Chinese, Cantonese, English, Japanese, Korean | Yes | No | [ModelScope (full)](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx) · [ModelScope (split‑embed)](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx) |
| sensevoice-small-wenetspeech-yue-int8-onnx | Offline | Cantonese, Chinese, English, Japanese, Korean | Yes | No | [ModelScope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-wenetspeech-yue-int8-onnx) |

## Model Introduction

### Purpose

Paraformer is an efficient non‑autoregressive end‑to‑end speech recognition framework proposed by the DAMO Academy Speech Team. This project provides a Paraformer Chinese speech recognition model trained on tens of thousands of hours of industrial‑grade annotated audio, delivering high accuracy. Typical use cases include voice input, voice navigation, and intelligent meeting transcription.

### Architecture

![Paraformer architecture diagram](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

The model consists of five parts: Encoder, Predictor, Sampler, Decoder, and Loss function.
- **Encoder**: Can use self‑attention, conformer, SAN‑M, etc.
- **Predictor**: Two‑layer FFN that predicts the number of target characters and extracts corresponding acoustic vectors.
- **Sampler**: No trainable parameters; fuses acoustic vectors and target vectors to produce semantic features.
- **Decoder**: Bi‑directional (auto‑regressive models are uni‑directional), enhancing context modeling.
- **Loss function**: Includes CE, MWER discriminative optimization, and MAE for the Predictor.

### Key Innovations

- **Predictor module**: Based on CIF (Continuous Integrate‑and‑Fire) for more accurate prediction of the number of target characters in speech.
- **Sampler**: Fuses acoustic vectors with target text vectors to produce semantic features; the bi‑directional Decoder leverages this for improved context modeling.
- **Negative‑sample MWER training criterion**.

### Further Reading

- [Paraformer Large Offline (ModelScope)](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
- [Paraformer Large Online (ModelScope)](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online)
- [SenseVoiceSmall (ModelScope)](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
- Paper: [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317)
- Paper summary (Chinese): [Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw)

## Reference

[1] https://github.com/alibaba-damo-academy/FunASR
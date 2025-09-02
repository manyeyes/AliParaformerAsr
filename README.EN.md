# AliParaformerAsr

## Overview  
AliParaformerAsr is a **speech recognition library** written in C#. Under the hood, it uses `Microsoft.ML.OnnxRuntime` for ONNX model decoding. It supports multiple environments including **net461+**, **net60+**, **netcoreapp3.1**, and **netstandard2.0+**, enabling cross-platform compilation and AOT (Ahead-of-Time) compilation. It is simple and easy to use.  


## Supported Models (ONNX)  

| Model Name | Type | Supported Languages | Punctuation | Timestamp | Download Link |
|------------|------|---------------------|-------------|-----------|---------------|
| paraformer-large-zh-en-onnx-offline | Non-streaming | Chinese, English | No | No | [huggingface](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx "huggingface"), [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-offline "modelscope") |
| paraformer-large-zh-en-timestamp-onnx-offline | Non-streaming | Chinese, English | No | Yes | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-timestamp-onnx-offline "modelscope") |
| paraformer-large-en-onnx-offline | Non-streaming | English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-en-onnx-offline "modelscope") |
| paraformer-large-zh-en-onnx-online | Streaming | Chinese, English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-online "modelscope") |
| paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 | Non-streaming | Chinese, Cantonese, English | No | Yes | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 "modelscope") |
| paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 | Non-streaming | Chinese, Cantonese, English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 "modelscope") |
| paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 | Streaming | Chinese, Cantonese, English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 "modelscope") |
| paraformer-seaco-large-zh-timestamp-onnx-offline | Non-streaming | Chinese, Hotword | No | Yes | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline "modelscope") |
| SenseVoiceSmall | Non-streaming | Chinese, Cantonese, English, Japanese, Korean | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx "modelscope"), [modelscope-split-embed](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx "modelscope-split-embed") |


## How to Use  
### 1. Clone the Project Source Code  
```bash
cd /path/to
git clone https://github.com/manyeyes/AliParaformerAsr.git
```

### 2. Download the Model to the Directory  
Download one of the models listed above to the directory: `/path/to/AliParaformerAsr/AliParaformerAsr.Examples`  
```bash
cd /path/to/AliParaformerAsr/AliParaformerAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[Model Name].git
```

### 3. Load the Project  
Open the project with Visual Studio 2022 (or another IDE).  

### 4. Configure Model File Properties  
Set the model files in the model directory to: **Copy to Output Directory -> Copy if newer**.  

### 5. Modify the Model Name in the Example Code  
Update the `modelName` variable in the example code to match your downloaded model directory name:  
- Non-streaming example: `OfflineRecognizer.cs`  
- Streaming example: `OnlineRecognizer.cs`  

### 6. Configuration Instructions (Refer to `asr.yaml`)  
The `asr.yaml` file contains configuration parameters for decoding. Most parameters do not need modification. The following parameter can be adjusted:  
- `use_itn: true`: Enables **inverse text normalization** (only supported in the SenseVoiceSmall model configuration).  

### 7. Run the Project  


## Invocation Method for Offline (Non-Streaming) Models  
### 1. Add Project Reference  
```csharp
using AliParaformerAsr;
```

### 2. Model Initialization and Configuration  
#### For Standard Paraformer Models  
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
string modelFilePath = applicationBase + "./" + modelName + "/model_quant.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";

AliParaformerAsr.OfflineRecognizer offlineRecognizer = new OfflineRecognizer(
    modelFilePath, 
    configFilePath, 
    mvnFilePath, 
    tokensFilePath
);
```

#### For SeACo-Paraformer Models  
1. Modify the `hotword.txt` file in the model directory to add custom hotwords (supports one Chinese term per line).  
2. Add two additional parameters in the code: `modelebFilePath` and `hotwordFilePath`.  

```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline";
string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
string modelebFilePath = applicationBase + "./" + modelName + "/model_eb.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string hotwordFilePath = applicationBase + "./" + modelName + "/hotword.txt";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";

OfflineRecognizer offlineRecognizer = new OfflineRecognizer(
    modelFilePath: modelFilePath, 
    configFilePath: configFilePath, 
    mvnFilePath, 
    tokensFilePath: tokensFilePath, 
    modelebFilePath: modelebFilePath, 
    hotwordFilePath: hotwordFilePath
);
```

### 3. Invocation  
```csharp
List<float[]> samples = new List<float[]>();
// WAV file to sample conversion is omitted here...
// For details, refer to the example code in AliParaformerAsr.Examples

List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
foreach (var sample in samples)
{
    AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}

List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```

### 4. Output Results  
```
欢迎大家来体验达摩院推出的语音识别模型

正是因为存在绝对正义所以我们接受现实的相对正义但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

非常的方便但是现在不同啊英国脱欧欧盟内部完善的产业链的红利人

he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉

after early nightfall the yellow lamps would light up here in there the squalid quarter of the broffles

elapsed_milliseconds:1502.8828125
total_duration:40525.6875
rtf:0.037084696280599808
end!
```

### 5. Output Timestamps  
To output character-level (Chinese) or word-level (English) timestamps, use a timestamp-supported ONNX model. The invocation method is the same as above.  

**Timestamp Unit**: ms  
```
he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉
he:[49,229]
must:[229,630]
be:[630,989]
home:[989,1350]
now:[1350,1589]
for:[1589,1829]
the:[1829,1949]
light:[1949,2270]
is:[2270,2490]
on:[2490,2770]
他:[2770,2970]
一:[2970,3129]
定:[3129,3310]
在:[3310,3490]
家:[3490,3689]
因:[3689,3790]
为:[3790,3990]
灯:[3990,4150]
亮:[4150,4430]
着:[4430,4830]
就:[4830,4970]
是:[4970,5089]
有:[5089,5230]
一:[5230,5350]
种:[5350,5950]
推:[5950,6230]
理:[6230,6430]
或:[6430,6569]
者:[6569,6770]
解:[6770,6990]
释:[6990,7170]
的:[7170,7529]
那:[7529,7710]
种:[7710,7870]
感:[7870,8010]
觉:[8010,8785]
```


## Invocation Method for Real-Time (Streaming) Models  
### 1. Add Project Reference  
```csharp
using AliParaformerAsr;
```

### 2. Model Initialization and Configuration  
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "paraformer-large-zh-en-onnx-online"; // Replace with your model name
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";

OnlineRecognizer onlineRecognizer = new OnlineRecognizer(
    encoderFilePath, 
    decoderFilePath, 
    configFilePath, 
    mvnFilePath, 
    tokensFilePath
);
```

### 3. Invocation  
```csharp
List<float[]> samples = new List<float[]>();
// WAV file to sample conversion is omitted here...
// For details, refer to the example code in AliParaformerAsr.Examples

// Batch processing example:
List<AliParaformerAsr.OnlineStream> streams = new List<AliParaformerAsr.OnlineStream>();
foreach (var sample in samples)
{
    AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<AliParaformerAsr.OnlineRecognizerResultEntity> batchResults = onlineRecognizer.GetResults(streams);

// Single-stream processing example:
AliParaformerAsr.OnlineStream singleStream = onlineRecognizer.CreateOnlineStream();
singleStream.AddSamples(sample); // Add audio samples
AliParaformerAsr.OnlineRecognizerResultEntity singleResult = onlineRecognizer.GetResult(singleStream);
```

### 4. Output Results  
```
正是

正是因为存

正是因为存在绝对正

正是因为存在绝对正义所以我

正是因为存在绝对正义所以我我接

正是因为存在绝对正义所以我我接受现实

正是因为存在绝对正义所以我我接受现实式相对生

正是因为存在绝对正义所以我我接受现实式相对生

正是因为存在绝对正义所以我我接受现实式相对生但是

正是因为存在绝对正义所以我我接受现实式相对生但是不要因

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

elapsed_milliseconds:1389.3125
total_duration:13052
rtf:0.10644441464909593
Hello, World!
```


## Related Projects  
1. **Voice Activity Detection (VAD)**  
   Solves the problem of reasonable segmentation for long audio files.  
   Project URL: [AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad")  

2. **Text Punctuation Prediction**  
   Solves the problem of missing punctuation in recognition results.  
   Project URL: [AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")  


## Additional Notes  
- **Test Cases**: Refer to `AliParaformerAsr.Examples` for complete usage examples.  
- **Test CPU**: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz  
- **Supported Platforms**:  
  - Windows 7 SP1 or later  
  - macOS 10.13 (High Sierra) or later, iOS  
  - Linux distributions (specific dependencies required; see the list of Linux distributions supported by .NET 6)  
  - Android (Android 5.0 (API 21) or later)  


## Model Introduction  
### Model Purpose  
Paraformer is an efficient non-autoregressive end-to-end speech recognition framework proposed by the Speech Team of Alibaba DAMO Academy. This project provides a general-purpose Chinese speech recognition model based on Paraformer, trained on industrial-grade annotated audio of tens of thousands of hours to ensure high general recognition performance. The model can be applied to scenarios such as speech input methods, voice navigation, and intelligent meeting minutes. **Accuracy**: High.  

### Model Structure  
![](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)  

As shown in the figure above, the Paraformer model consists of five components: **Encoder**, **Predictor**, **Sampler**, **Decoder**, and **Loss Function**:  
- The Encoder can adopt different network structures (e.g., self-attention, Conformer, SAN-M).  
- The Predictor is a two-layer FFN (Feed-Forward Network) that predicts the number of target characters and extracts acoustic vectors corresponding to the target characters.  
- The Sampler is a module without trainable parameters. It generates semantic feature vectors based on input acoustic vectors and target vectors.  
- The Decoder uses bidirectional modeling (unlike autoregressive models, which use unidirectional modeling) and has a structure similar to autoregressive models.  
- The Loss Function includes cross-entropy (CE), MWER (Minimum Word Error Rate) discriminative optimization objectives, and the Predictor optimization objective MAE (Mean Absolute Error).  

### Key Core Features  
1. **Predictor Module**: Uses a **Continuous Integrate-and-Fire (CIF)-based Predictor** to extract acoustic feature vectors corresponding to target characters, enabling more accurate prediction of the number of target characters in speech.  
2. **Sampler**: Transforms acoustic feature vectors and target character vectors into semantic feature vectors through sampling, and cooperates with the bidirectional Decoder to enhance the model’s ability to model context.  
3. **MWER Training Criterion**: Uses MWER (Minimum Word Error Rate) training based on negative sample sampling.  

### More Detailed Resources  
- [paraformer-large-offline (Non-streaming)](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch "paraformer-large-offline (Non-streaming)")  
- [paraformer-large-online (Streaming)](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online "paraformer-large-online (Streaming)")  
- [SenseVoiceSmall (Non-streaming)](https://www.modelscope.cn/models/iic/SenseVoiceSmall "SenseVoiceSmall (Non-streaming)")  
- Paper: [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317 "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition")  
- Paper Interpretation: [Paraformer: A High-Accuracy, High-Efficiency Single-Pass Non-Autoregressive End-to-End Speech Recognition Model](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw "Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型")  


## References  
[1] https://github.com/alibaba-damo-academy/FunASR
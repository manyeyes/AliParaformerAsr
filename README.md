# AliParaformerAsr

## 简介

**AliParaformerAsr** 是一个使用 C# 编写的语音识别库，底层调用 `Microsoft.ML.OnnxRuntime` 对 ONNX 模型进行解码。支持 .NET Framework 4.6.1+、.NET 6.0+、.NET Core 3.1 及 .NET Standard 2.0+ 等多种运行环境，支持跨平台编译及 AOT 编译，使用简单方便。

## 如何运行示例项目

### 1. 克隆项目到本地

```bash
cd /path/to
git clone https://github.com/manyeyes/AliParaformerAsr.git
```

### 2. 下载模型（可选，示例运行时会自动下载）

```bash
cd /path/to/AliParaformerAsr/AliParaformerAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[模型名称].git
```

### 3. 使用 IDE 加载工程

推荐使用 Visual Studio 2022（或其它支持 .NET 的 IDE）。

### 4. 运行示例项目

- **AliParaformerAsr.Examples**：控制台/桌面端示例，演示离线转写、实时识别等基础功能。
- **MauiApp1**：基于 .NET MAUI 的跨平台示例，支持 Android、iOS、Windows 等设备。

### 5. 配置说明

配置文件为 `asr.yaml`，大部分参数无需修改。常用可修改参数：

| 参数       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| `use_itn`  | 设置为 `true` 时（在 SenseVoiceSmall 模型中）可开启逆文本正则化。 |

## 如何在代码中调用

### 离线（非流式）模型

#### 1. 添加引用

```csharp
using AliParaformerAsr;
```

#### 2. 初始化模型

**普通 Paraformer 模型：**

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

**SeACo-Paraformer 模型（支持热词）：**

1. 在模型目录中编辑 `hotword.txt` 文件，每行一个中文热词。
2. 代码中需额外指定 `modelebFilePath` 和 `hotwordFilePath`。

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

#### 3. 调用识别

```csharp
List<float[]> samples = new List<float[]>();
// 此处省略从 wav 文件读取 samples 的代码（可参考示例项目）

List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}

List<Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```

#### 4. 输出示例

```
欢迎大家来体验达摩院推出的语音识别模型
非常的方便但是现在不同啊英国脱欧欧盟内部完善的产业链的红利人
he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉
elapsed_milliseconds:1502.8828125
total_duration:40525.6875
rtf:0.037084696280599808
end!
```

### 实时（流式）模型

#### 1. 添加引用

```csharp
using AliParaformerAsr;
```

#### 2. 初始化模型

```csharp
string encoderFilePath = Path.Combine(applicationBase, modelName, "encoder.int8.onnx");
string decoderFilePath = Path.Combine(applicationBase, modelName, "decoder.int8.onnx");
string configFilePath = Path.Combine(applicationBase, modelName, "asr.yaml");
string mvnFilePath = Path.Combine(applicationBase, modelName, "am.mvn");
string tokensFilePath = Path.Combine(applicationBase, modelName, "tokens.txt");

OnlineRecognizer onlineRecognizer = new OnlineRecognizer(
    encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
```

#### 3. 调用识别

**批处理：**

```csharp
List<float[]> samples = new List<float[]>();
// 读取 samples...

List<OnlineStream> streams = new List<OnlineStream>();
foreach (var sample in samples)
{
    OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OnlineRecognizerResultEntity> results = onlineRecognizer.GetResults(streams);
```

**单条处理：**

```csharp
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
stream.AddSamples(sample);
OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
```

#### 4. 输出示例

```
正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界
elapsed_milliseconds:1389.3125
total_duration:13052
rtf:0.10644441464909593
Hello, World!
```

## 相关工程

| 项目名称                     | 说明                               | 地址                                                                                  |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| AliFsmnVad                   | 语音端点检测，解决长音频合理切分   | [GitHub](https://github.com/manyeyes/AliFsmnVad)                                       |
| AliCTTransformerPunc         | 文本标点预测，为识别结果添加标点   | [GitHub](https://github.com/manyeyes/AliCTTransformerPunc)                             |

## 其他说明

- **测试用例**：`AliParaformerAsr.Examples`
- **测试 CPU**：Intel(R) Core(TM) i7-10750H @ 2.60GHz
- **支持平台**：
  - Windows 7 SP1 或更高版本
  - macOS 10.13 (High Sierra) 或更高版本（含 iOS）
  - Linux 发行版（需符合 .NET 6 支持列表）
  - Android 5.0 (API 21) 或更高版本

## 模型下载（ONNX 格式）

| 模型名称                                                                   | 类型   | 支持语言                 | 标点 | 时间戳 | 下载地址                                                                                                                                                                                                                  |
| -------------------------------------------------------------------------- | ------ | ------------------------ | ---- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| paraformer-large-zh-en-onnx-offline                                        | 非流式 | 中文、英文               | 否   | 否     | [🤗 HuggingFace](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx) · [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-offline)           |
| paraformer-large-zh-en-timestamp-onnx-offline                              | 非流式 | 中文、英文               | 否   | 是     | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-timestamp-onnx-offline)                                                                                                                     |
| paraformer-large-en-onnx-offline                                           | 非流式 | 英文                     | 否   | 否     | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-en-onnx-offline)                                                                                                                                 |
| paraformer-large-zh-en-onnx-online                                         | 流式   | 中文、英文               | 否   | 否     | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-online)                                                                                                                               |
| paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805      | 非流式 | 中文、粤语、英文         | 否   | 是     | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805)                                                                                             |
| paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805                | 非流式 | 中文、粤语、英文         | 否   | 否     | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805)                                                                                                       |
| paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208                 | 流式   | 中文、粤语、英文         | 否   | 否     | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208)                                                                                                        |
| paraformer-seaco-large-zh-timestamp-onnx-offline                           | 非流式 | 中文（热词）             | 否   | 是     | [ModelScope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline)                                                                                                                  |
| SenseVoiceSmall                                                            | 非流式 | 中文、粤语、英文、日、韩 | 是   | 否     | [ModelScope（完整）](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx) · [ModelScope（split-embed）](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx)                         |
| sensevoice-small-wenetspeech-yue-int8-onnx                                 | 非流式 | 粤语、中文、英文、日、韩 | 是   | 否     | [ModelScope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-wenetspeech-yue-int8-onnx)                                                                                                                        |

## 模型介绍

### 模型用途

Paraformer 是达摩院语音团队提出的高效非自回归端到端语音识别框架。本项目为 Paraformer 中文通用语音识别模型，采用数万小时工业级标注音频训练，识别准确率高，适用于语音输入法、语音导航、智能会议纪要等场景。

### 模型结构

![Paraformer 模型结构图](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

模型由 Encoder、Predictor、Sampler、Decoder 与 Loss function 五部分组成：
- **Encoder**：可采用 self-attention、conformer、SAN-M 等结构。
- **Predictor**：两层 FFN，预测目标文字个数并抽取对应声学向量。
- **Sampler**：无可学习参数，将声学向量与目标向量融合为含语义的特征向量。
- **Decoder**：双向建模（自回归为单向），增强上下文建模能力。
- **Loss function**：包含交叉熵（CE）、MWER 区分性优化目标以及 Predictor 的 MAE 优化目标。

### 核心技术

- **Predictor 模块**：基于 CIF（Continuous integrate-and-fire）机制，更准确预测语音中目标文字的个数。
- **Sampler**：通过采样使声学向量与目标文字向量融合为带语义的特征，配合双向 Decoder 提升上下文建模能力。
- **基于负样本采样的 MWER 训练准则**。

### 进一步资料

- [Paraformer 大模型离线版（ModelScope）](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
- [Paraformer 大模型在线版（ModelScope）](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online)
- [SenseVoiceSmall（ModelScope）](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
- 论文：[Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317)
- 论文解读：[Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw)

## 引用参考

[1] https://github.com/alibaba-damo-academy/FunASR
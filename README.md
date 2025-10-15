# 📢 项目迁移重要通知

为便于统一管理和维护，**AliParaformerAsr 项目已完整迁移至新仓库**：

🔗 **新仓库地址**  
[ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech 主仓库")  

此后，所有代码更新、问题反馈及功能维护将统一在 ManySpeech 仓库进行。请各位用户和开发者前往新仓库获取最新资源。


### ⚠️ 迁移后关键变更
- **命名空间调整**  
  由原 `AliParaformerAsr` 变更为 `ManySpeech.AliParaformerAsr`，请在集成时注意更新引用。

- **资源升级**  
  新仓库提供更丰富的使用示例、更详细的 API 文档及持续的功能迭代。

  感谢您的理解与支持!

----------------------------------------------------------------

# AliParaformerAsr

## 简介：

AliParaformerAsr是一个使用C#编写的“语音识别”库，底层调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持 net461+、net60+、netcoreapp3.1 及 netstandard2.0+ 等多种环境，支持跨平台编译，支持AOT编译。使用简单方便。

## 如何运行示例项目

#### 1.克隆项目到本地
```bash
cd /path/to
git clone https://github.com/manyeyes/AliParaformerAsr.git
``` 
#### 2.下载上述列表中的模型到本地目录备用（运行示例将自动下载模型，此步骤可跳过）
```bash
cd /path/to/AliParaformerAsr/AliParaformerAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[模型名称].git
```
#### 3.使用vs2022(或其他IDE)加载工程，
#### 4.运行 AliParaformerAsr.Examples 项目
AliParaformerAsr.Examples 是控制台 / 桌面端示例项目，用于演示语音识别的基础功能（如离线转写、实时识别）。
#### 5.运行 MauiApp1 项目
MauiApp1 是基于 .NET MAUI 开发的跨平台项目，支持在 Android、iOS、Windows 等设备上运行语音识别功能。
#### 6.配置说明（参考：asr.yaml文件）：
用于解码的asr.yaml配置参数，大部分不需要修改。
可修改的参数：
use_itn: true（在sensevoicesmall的配置中开启之后，可实现逆文本正则化。）

## 如何在代码中调用

### 离线（非流式）模型调用方法：

#### 1.添加项目引用
using AliParaformerAsr;

#### 2.模型初始化和配置
paraformer模型调用方式：

```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
string modelFilePath = applicationBase + "./"+ modelName + "/model_quant.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
AliParaformerAsr.OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
SeACo-paraformer模型调用方式：
1.在模型目录中修改hotword.txt文件，添加自定义热词（目前支持“每一行一个中文词汇”的格式）
2.在代码中新增参数：modelebFilePath, hotwordFilePath
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline";
string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
string modelebFilePath = applicationBase + "./" + modelName + "/model_eb.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string hotwordFilePath = applicationBase + "./" + modelName + "/hotword.txt";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath, tokensFilePath: tokensFilePath, modelebFilePath: modelebFilePath, hotwordFilePath: hotwordFilePath);
```
#### 3.调用
```csharp
List<float[]> samples = new List<float[]>();
//这里省略wav文件转samples...
//具体参考示例（AliParaformerAsr.Examples）代码
List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
foreach (var sample in samples)
{
    AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```
#### 4.输出结果：
```
欢迎大家来体验达摩院推出的语音识别模型

非常的方便但是现在不同啊英国脱欧欧盟内部完善的产业链的红利人

he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉

elapsed_milliseconds:1502.8828125
total_duration:40525.6875
rtf:0.037084696280599808
end!
```

## 实时（流式）模型调用方法：

#### 1.添加项目引用
using AliParaformerAsr;

#### 2.模型初始化和配置
```csharp
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
#### 3.调用
```csharp
List<float[]> samples = new List<float[]>();
//这里省略wav文件转samples...
//这里省略细节，以下是批处理示意代码：
List<AliParaformerAsr.OnlineStream> streams = new List<AliParaformerAsr.OnlineStream>();
AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
foreach (var sample in samples)
{
    AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<AliParaformerAsr.OnlineRecognizerResultEntity> results = onlineRecognizer.GetResults(streams);
//单处理，只需构建一个stream:
AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
stream.AddSamples(sample);
AliParaformerAsr.OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
//具体参考示例（AliParaformerAsr.Examples）代码
```

#### 4.输出结果
```

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

elapsed_milliseconds:1389.3125
total_duration:13052
rtf:0.10644441464909593
Hello, World!
```


#### 相关工程：
* 语音端点检测，解决长音频合理切分的问题，项目地址：[AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 
* 文本标点预测，解决识别结果没有标点的问题，项目地址：[AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")

#### 其他说明：

测试用例：AliParaformerAsr.Examples。
测试CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
支持平台：
Windows 7 SP1或更高版本,
macOS 10.13 (High Sierra) 或更高版本,ios等，
Linux 发行版（需要特定的依赖关系，详见.NET 6支持的Linux发行版列表），
Android（Android 5.0 (API 21) 或更高版本）。

#### 模型下载（ONNX 模型）

| 模型名称  |  类型 | 支持语言  | 标点  |  时间戳 | 下载地址  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  paraformer-large-zh-en-onnx-offline | 非流式  | 中文、英文  |  否 | 否  | [huggingface](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx "huggingface"),  [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-offline "modelscope") |
|  paraformer-large-zh-en-timestamp-onnx-offline | 非流式  | 中文、英文  |  否 | 是  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-timestamp-onnx-offline "modelscope") |
|  paraformer-large-en-onnx-offline | 非流式 | 英文 |  否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-en-onnx-offline "modelscope")  |
|  paraformer-large-zh-en-onnx-online | 流式 | 中文、英文 |  否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-online "modelscope")  |
|  paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 | 非流式 | 中文、粤语、英文 |  否  | 是  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 "modelscope")  |
|  paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 |  非流式 | 中文、粤语、英文 | 否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 "modelscope") |
|  paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 |  流式 | 中文、粤语、英文 | 否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 "modelscope") |
|  paraformer-seaco-large-zh-timestamp-onnx-offline |  非流式 | 中文、热词 | 否  | 是  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline "modelscope") |
|  SenseVoiceSmall |  非流式 | 中文、粤语、英文、日语、韩语 | 是  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx "modelscope"), [modelscope-split-embed](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx "modelscope-split-embed") |
|  sensevoice-small-wenetspeech-yue-int8-onnx |  非流式 | 粤语、中文、英文、日语、韩语 | 是  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-wenetspeech-yue-int8-onnx "modelscope") |

## 模型介绍：

#### 模型用途：
Paraformer是达摩院语音团队提出的一种高效的非自回归端到端语音识别框架。本项目为Paraformer中文通用语音识别模型，采用工业级数万小时的标注音频进行模型训练，保证了模型的通用识别效果。模型可以被应用于语音输入法、语音导航、智能会议纪要等场景。准确率：高。

#### 模型结构：
![](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

Paraformer模型结构如上图所示，由 Encoder、Predictor、Sampler、Decoder 与 Loss function 五部分组成。Encoder可以采用不同的网络结构，例如self-attention，conformer，SAN-M等。Predictor 为两层FFN，预测目标文字个数以及抽取目标文字对应的声学向量。Sampler 为无可学习参数模块，依据输入的声学向量和目标向量，生产含有语义的特征向量。Decoder 结构与自回归模型类似，为双向建模（自回归为单向建模）。Loss function 部分，除了交叉熵（CE）与 MWER 区分性优化目标，还包括了 Predictor 优化目标 MAE。

#### 主要核心点：
Predictor 模块：基于 Continuous integrate-and-fire (CIF) 的 预测器 (Predictor) 来抽取目标文字对应的声学特征向量，可以更加准确的预测语音中目标文字个数。
Sampler：通过采样，将声学特征向量与目标文字向量变换成含有语义信息的特征向量，配合双向的 Decoder 来增强模型对于上下文的建模能力。
基于负样本采样的 MWER 训练准则。

#### 更详细的资料：
* [paraformer-large-offline（非流式）](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch "paraformer-large-offline（非流式）")
* [paraformer-large-online（流式）](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online "paraformer-large-online（流式）")
* [SenseVoiceSmall（非流式）](https://www.modelscope.cn/models/iic/SenseVoiceSmall "SenseVoiceSmall（非流式）")
* 论文： [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317 "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition")
* 论文解读：[Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw "Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型")

引用参考
----------
[1] https://github.com/alibaba-damo-academy/FunASR


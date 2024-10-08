# AliParaformerAsr

##### 简介：

**AliParaformerAsr是一个使用C#编写的“语音识别”库，底层调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持框架.Net6.0+，支持跨平台编译，支持AOT编译。使用简单方便。**

##### 支持的模型（ONNX）

| 模型名称  |  类型 | 支持语言  | 标点  |  时间戳 | 下载地址  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  paraformer-large-zh-en-onnx-offline | 非流式  | 中文、英文  |  否 | 否  | [huggingface](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx "huggingface"),  [modelscope](https://www.modelscope.cn/models/manyeyes/aliparaformerasr-large-zh-en-onnx-offline "modelscope") |
|  paraformer-large-zh-en-timestamp-onnx-offline | 非流式  | 中文、英文  |  否 | 是  | [modelscope](https://www.modelscope.cn/models/manyeyes/aliparaformerasr-large-zh-en-timestamp-onnx-offline "modelscope") |
|  paraformer-large-en-onnx-offline | 非流式 | 英文 |  否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/aliparaformerasr-large-en-onnx-offline "modelscope")  |
|  paraformer-large-zh-en-onnx-online | 流式 | 中文、英文 |  否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/aliparaformerasr-large-zh-en-onnx-online "modelscope")  |
|  paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 | 非流式 | 中文、粤语、英文 |  否  | 是  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 "modelscope")  |
|  paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 |  非流式 | 中文、粤语、英文 | 否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 "modelscope") |
|  paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 |  流式 | 中文、粤语、英文 | 否  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 "modelscope") |
|  paraformer-seaco-large-zh-timestamp-onnx-offline |  非流式 | 中文、热词 | 否  | 是  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline "modelscope") |
|  SenseVoiceSmall |  非流式 | 中文、粤语、英文、日语、韩语 | 是  | 否  | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx "modelscope"), [modelscope-split-embed](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx "modelscope-split-embed") |


##### 如何使用
###### 1.克隆项目源码
```bash
cd /path/to
git clone https://github.com/manyeyes/AliParaformerAsr.git
```
###### 2.下载上述列表中的模型到目录：/path/to/AliParaformerAsr/AliParaformerAsr.Examples
```bash
cd /path/to/AliParaformerAsr/AliParaformerAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[模型名称].git
```
###### 3.使用vs2022(或其他IDE)加载工程，
###### 4.将模型目录中的文件设置为：复制到输出目录->如果较新则复制
###### 5.修改示例中代码：string modelName =[模型目录名]
非流式示例：OfflineRecognizer.cs
流式示例：OnlineRecognizer.cs
###### 6.配置说明（参考：asr.yaml文件）：
用于解码的asr.yaml配置参数，大部分不需要修改。
可修改的参数：
use_itn: true（在sensevoicesmall的配置中开启之后，可实现逆文本正则化。）
###### 7.运行项目

## 离线（非流式）模型调用方法：

###### 1.添加项目引用
using AliParaformerAsr;

###### 2.模型初始化和配置
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
###### 3.调用
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
###### 4.输出结果：
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

###### 5.输出时间戳
如需输出字级（中文）或单词级（英文）时间戳，使用带时间戳的onnx模型即可，其他调用方式与上述一致。
时间戳单位：ms
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

## 实时（流式）模型调用方法：

###### 1.添加项目引用
using AliParaformerAsr;

###### 2.模型初始化和配置
```csharp
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
###### 3.调用
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

###### 4.输出结果
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


###### 相关工程：
* 语音端点检测，解决长音频合理切分的问题，项目地址：[AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 
* 文本标点预测，解决识别结果没有标点的问题，项目地址：[AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")

###### 其他说明：

测试用例：AliParaformerAsr.Examples。
测试CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
支持平台：
Windows 7 SP1或更高版本,
macOS 10.13 (High Sierra) 或更高版本,ios等，
Linux 发行版（需要特定的依赖关系，详见.NET 6支持的Linux发行版列表），
Android（Android 5.0 (API 21) 或更高版本）。

## 模型介绍：

##### 模型用途：
Paraformer是达摩院语音团队提出的一种高效的非自回归端到端语音识别框架。本项目为Paraformer中文通用语音识别模型，采用工业级数万小时的标注音频进行模型训练，保证了模型的通用识别效果。模型可以被应用于语音输入法、语音导航、智能会议纪要等场景。准确率：高。

##### 模型结构：
![](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

Paraformer模型结构如上图所示，由 Encoder、Predictor、Sampler、Decoder 与 Loss function 五部分组成。Encoder可以采用不同的网络结构，例如self-attention，conformer，SAN-M等。Predictor 为两层FFN，预测目标文字个数以及抽取目标文字对应的声学向量。Sampler 为无可学习参数模块，依据输入的声学向量和目标向量，生产含有语义的特征向量。Decoder 结构与自回归模型类似，为双向建模（自回归为单向建模）。Loss function 部分，除了交叉熵（CE）与 MWER 区分性优化目标，还包括了 Predictor 优化目标 MAE。

##### 主要核心点：
Predictor 模块：基于 Continuous integrate-and-fire (CIF) 的 预测器 (Predictor) 来抽取目标文字对应的声学特征向量，可以更加准确的预测语音中目标文字个数。
Sampler：通过采样，将声学特征向量与目标文字向量变换成含有语义信息的特征向量，配合双向的 Decoder 来增强模型对于上下文的建模能力。
基于负样本采样的 MWER 训练准则。

##### 更详细的资料：
* [paraformer-large-offline（非流式）](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch "paraformer-large-offline（非流式）")
* [paraformer-large-online（流式）](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online "paraformer-large-online（流式）")
* [SenseVoiceSmall（非流式）](https://www.modelscope.cn/models/iic/SenseVoiceSmall "SenseVoiceSmall（非流式）")
* 论文： [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317 "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition")
* 论文解读：[Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw "Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型")

引用参考
----------
[1] https://github.com/alibaba-damo-academy/FunASR

[2] https://github.com/naudio/NAudio


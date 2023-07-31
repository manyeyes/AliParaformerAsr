# Paraformer-large模型介绍
### 使用方法
下载模型
```shell
git clone https://github.com/manyeyes/AliParaformerAsr.git
cd /AliParaformerAsr/AliParaformerAsr.Examples
git clone https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx.git
```
编辑 Program.cs 中的 modelName 值：
```csharp
string modelName="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
//[fp32模型/int8模型]，二选一
//fp32模型
//string modelFilePath = applicationBase + "./"+ modelName + "/model.onnx";
//int8模型
string modelFilePath = applicationBase + "./"+ modelName + "/model_quant.onnx";
```
将文件夹 speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx 中的文件属性-》复制到输出目录-》如果较新则复制。
**或者**
编辑 AliParaformerAsr.Examples.csproj ，添加
```xml
<ItemGroup>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\am.mvn">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\asr.yaml">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\example\0.wav">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\example\1.wav">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\example\2.wav">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\example\3.wav">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\example\4.wav">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\model_quant.onnx">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\model.onnx">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Update="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\tokens.txt">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
  </ItemGroup>
```

启动 AliParaformerAsr.Examples


## 通过modelscope了解更多
### 原模型
- pytorch
https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
- onnx
https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/summary

### Highlights
- 热词版本：[Paraformer-large热词版模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary)支持热词定制功能，基于提供的热词列表进行激励增强，提升热词的召回率和准确率。
- 长音频版本：[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。
using AliParaformerAsr;
using FluentAssertions;
using System;
using System.Reflection;
using Xunit;

namespace AliParaformerAsr.Tests;

/// <summary>
/// 测试 OfflineStream 核心功能：添加采样、设置热词、状态验证
/// </summary>
public class OfflineStreamTests : IDisposable
{
    // 依赖 OfflineRecognizer 创建流（需先初始化识别器）
    private readonly OfflineRecognizer _recognizer;
    // 待测试的流实例
    private readonly OfflineStream _stream;

    public OfflineStreamTests()
    {
        // 1. 初始化基础识别器（复用前一个测试的模拟模型路径逻辑）
        var mockModelDir = Path.Combine(
            Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!,
            "TestResources", "MockModels"
        );
        var mockTokensPath = Path.Combine(mockModelDir, "tokens.txt");

        // 2. 确保 tokens.txt 存在（避免初始化失败）
        if (!File.Exists(mockTokensPath))
        {
            File.WriteAllText(mockTokensPath, "<s>\n</s>\n你\n好");
        }

        // 3. 创建识别器并生成流
        _recognizer = new OfflineRecognizer(
            modelFilePath: Path.Combine(mockModelDir, "model.int8.onnx"),
            configFilePath: Path.Combine(mockModelDir, "asr.yaml"),
            mvnFilePath: Path.Combine(mockModelDir, "am.mvn"),
            tokensFilePath: mockTokensPath,
            modelebFilePath: Path.Combine(mockModelDir, "model_eb.int8.onnx"),
            hotwordFilePath: "",
            threadsNum: 2
        );
        _stream = _recognizer.CreateOfflineStream();
    }

    /// <summary>
    /// 测试1：添加合法采样（非空 float 数组）→ 无异常
    /// </summary>
    [Fact]
    public void OfflineStream_AddSamples_WithValidSamples_ShouldNotThrow()
    {
        // Arrange：生成合法采样（1000个 float 元素）
        var validSamples = new float[1000];
        Array.Fill(validSamples, 0.1f); // 非静音采样（模拟真实音频）

        // Act：添加采样（无异常即通过）
        Action addAction = () => _stream.AddSamples(validSamples);

        // Assert：无异常抛出
        addAction.Should().NotThrow();
    }

    /// <summary>
    /// 测试2：添加空采样数组 → 应抛出 ArgumentNullException
    /// </summary>
    [Fact]
    public void OfflineStream_AddSamples_WithNullSamples_ShouldThrowArgumentNullException()
    {
        // Arrange：空采样数组
        float[]? nullSamples = null;

        // Act + Assert：添加空采样 → 抛出空参数异常
        Action addAction = () => _stream.AddSamples(nullSamples!);
        addAction.Should().Throw<ArgumentNullException>()
            .WithParameterName("samples"); // 异常应指向 "samples" 参数
    }

    /// <summary>
    /// 测试3：设置热词（合法字符串）→ 无异常（热词生效与否不依赖真实识别）
    /// </summary>
    [Fact]
    public void OfflineStream_SetHotwords_WithValidText_ShouldNotThrow()
    {
        // Arrange：合法热词（如“魔搭”“语音识别”）
        var hotwords = "魔搭,语音识别,人工智能";

        // Act：设置热词（核心库若支持 Hotwords 属性，直接赋值）
        Action setHotwordAction = () => _stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(modelBasePath, modelName, "tokens.txt"), new string[] { "魔搭", "语音识别", "人工智能" });

        // Assert：无异常抛出（热词格式合法）
        setHotwordAction.Should().NotThrow();
        _stream.Hotwords.Should().BeEquivalentTo(hotwords); // 验证热词已设置
    }

    /// <summary>
    /// 测试4：设置空热词 → 应允许（清空热词）
    /// </summary>
    [Fact]
    public void OfflineStream_SetHotwords_WithEmptyText_ShouldClearHotwords()
    {
        // Arrange：先设置非空热词，再设为空
        _stream.Hotwords = "魔搭";
        var emptyHotwords = "";

        // Act：设置空热词
        _stream.Hotwords = emptyHotwords;

        // Assert：热词已清空
        _stream.Hotwords.Should().BeEmpty();
    }

    /// <summary>
    /// 测试清理：释放识别器和流资源
    /// </summary>
    public void Dispose()
    {
        _stream.Dispose(); // 若流支持 Dispose，显式释放
        _recognizer.Dispose();
    }
}
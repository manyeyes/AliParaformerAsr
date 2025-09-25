using AliParaformerAsr;
using AliParaformerAsr.Tests.Utils;
using AliParaformerAsr.Model;
using FluentAssertions;
using Microsoft.VisualStudio.TestPlatform.PlatformAbstractions.Interfaces;
using Moq;
using System.IO;
using System.Reflection;
using Xunit;

namespace AliParaformerAsr.Tests;

/// <summary>
/// 测试 OfflineRecognizer 核心功能：初始化、创建流、获取结果、释放资源
/// </summary>
public class OfflineRecognizerTests : IDisposable
{
    // 测试资源路径：模拟模型文件所在目录
    private string _ModelDir;
    // 模拟模型文件路径（预构建合法路径）
    private string _modelFilePath;
    private string _configFilePath;
    private string _mvnFilePath;
    private string _tokensFilePath;
    private string _modelebFilePath;
    private string _hotwordFilePath;
    private int _threadsNum = 2;

    // 待测试的识别器实例
    private OfflineRecognizer? _recognizer;

    public OfflineRecognizerTests()
    {
        string modelBasePath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, "TestResources");
        string modelName = "sensevoice-small-int8-onnx";
        // 1. 初始化测试资源路径（获取 TestResources/MockModels 目录）
        Directory.CreateDirectory(modelBasePath);
        _ModelDir = Path.Combine(
            modelBasePath, modelName
        );

        //// 2.下载模型
        //GitHelper gitHelper = new GitHelper();
        //Task.Run(() => gitHelper.ProcessCloneModel(modelBasePath, modelName));

        // 3. 构建模拟模型文件的完整路径（确保文件存在）
        _modelFilePath = Path.Combine(_ModelDir, "model.int8.onnx");
        _configFilePath = Path.Combine(_ModelDir, "asr.yaml");
        _mvnFilePath = Path.Combine(_ModelDir, "am.mvn");
        _tokensFilePath = Path.Combine(_ModelDir, "tokens.txt");
        _modelebFilePath = Path.Combine(_ModelDir, "model_eb.int8.onnx");

        // 4. 确保模拟文件存在（若不存在则创建空文件，避免路径错误）
        EnsureMockFilesExist();
    }

    /// <summary>
    /// 确保所有模拟模型文件存在（避免 FileNotFoundException）
    /// </summary>
    private void EnsureMockFilesExist(string modelAccuracy = "int8")
    {
        try
        {
            string folderPath = _ModelDir;
            // 1. Check if the folder exists
            if (!Directory.Exists(folderPath))
            {
                Console.WriteLine($"Error: folder does not exist - {folderPath}");
                return;
            }
            // 2. Obtain the file names and destination paths of all files
            // (calculate the paths in advance to avoid duplicate concatenation)
            var fileInfos = Directory.GetFiles(folderPath)
                .Select(filePath => new
                {
                    FileName = Path.GetFileName(filePath),
                    // Recommend using Path. Combine to handle paths (automatically adapt system separators)
                    TargetPath = Path.Combine(folderPath, Path.GetFileName(filePath))
                    // If it is necessary to strictly maintain the original splicing method, it can be replaced with:
                    // TargetPath = $"{modelBasePath}/./{modelName}/{Path.GetFileName(filePath)}"
                })
                .ToList();

            // Process model path (priority: containing modelAccuracy>last one that matches prefix)
            var modelCandidates = fileInfos
                .Where(f => f.FileName.StartsWith("model") && !f.FileName.Contains("_eb"))
                .ToList();
            if (modelCandidates.Any())
            {
                // Prioritize selecting files that contain the specified model accuracy
                var preferredModel = modelCandidates
                    .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                _modelFilePath = preferredModel?.TargetPath ?? modelCandidates.Last().TargetPath;
            }

            // Process modeleb path
            var modelebCandidates = fileInfos
                .Where(f => f.FileName.StartsWith("model_eb"))
                .ToList();
            if (modelebCandidates.Any())
            {
                var preferredModeleb = modelebCandidates
                    .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                _modelebFilePath = preferredModeleb?.TargetPath ?? modelebCandidates.Last().TargetPath;
            }

            // Process config paths (take the last one that matches the prefix)
            _configFilePath = fileInfos
                .LastOrDefault(f => f.FileName.StartsWith("asr") && (f.FileName.EndsWith(".yaml") || f.FileName.EndsWith(".json")))
                ?.TargetPath ?? "";

            // Process mvn paths (take the last one that matches the prefix)
            _mvnFilePath = fileInfos
                .LastOrDefault(f => f.FileName.StartsWith("am") && f.FileName.EndsWith(".mvn"))
                ?.TargetPath ?? "";

            // Process token paths (take the last one that matches the prefix)
            _tokensFilePath = fileInfos
                .LastOrDefault(f => f.FileName.StartsWith("tokens") && f.FileName.EndsWith(".txt"))
                ?.TargetPath ?? "";

            // Process hotword paths (take the last one that matches the prefix)
            _hotwordFilePath = fileInfos
                .LastOrDefault(f => f.FileName.StartsWith("hotword") && f.FileName.EndsWith(".txt"))
                ?.TargetPath ?? "";
        }
        catch (UnauthorizedAccessException)
        {
            Console.WriteLine($"Error: No permission to access this folder");
        }
        catch (PathTooLongException)
        {
            Console.WriteLine($"Error: File path too long");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error occurred: {ex}");
        }
    }

    //// 测试异步工厂方法：确保模型克隆完成且对象初始化正确
    //[Fact]
    //public async Task CreateAsync_ShouldCloneModelAndInitializePath()
    //{
    //    // Arrange：准备测试路径和参数
    //    string modelBasePath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, "TestResources");
    //    string modelName = "sensevoice-small-int8-onnx";
    //    string expectedModelPath = Path.Combine(modelBasePath, modelName);

    //    // 清理可能的残留文件（确保测试独立性）
    //    //if (Directory.Exists(expectedModelPath))
    //    //    Directory.Delete(expectedModelPath, recursive: true);

    //    // 2.下载模型
    //    GitHelper gitHelper = new GitHelper();
    //    await Task.Run(() => gitHelper.ProcessCloneModel(modelBasePath, modelName));

    //    // Assert：验证结果
    //    // 1. 模型目录应被创建（克隆成功）
    //    Assert.True(Directory.Exists(expectedModelPath), "模型克隆失败，目录未创建");
    //}

    /// <summary>
    /// 测试1：合法参数初始化 OfflineRecognizer → 应成功（识别器非空）
    /// </summary>
    [Fact]
    public void OfflineRecognizer_Init_WithValidParams_ShouldReturnNonNull()
    {
        //if (string.IsNullOrEmpty(_modelFilePath) || string.IsNullOrEmpty(_tokensFilePath))
        //{
        //    return;
        //}

        // Act：创建识别器实例
        InitRecognizer();

        // Assert：识别器非空，初始化成功
        _recognizer.Should().NotBeNull();
    }

    /// <summary>
    /// 测试2：缺失关键文件（如 tokens.txt）→ 初始化应抛出异常
    /// </summary>
    [Fact]
    public void OfflineRecognizer_Init_WithMissingTokensFile_ShouldThrowException()
    {
        // Arrange：删除 tokens.txt（模拟文件缺失）
        //if (File.Exists(_mockTokensPath))
        //{
        //    File.Delete(_mockTokensPath);
        //}

        _tokensFilePath = "";

        // Act + Assert：初始化应抛出异常（关键文件缺失）
        Action initAction = () => new OfflineRecognizer(
            modelFilePath: _modelFilePath,
            configFilePath: _configFilePath,
            mvnFilePath: _mvnFilePath,
            tokensFilePath: _tokensFilePath,
            modelebFilePath: _modelebFilePath,
            hotwordFilePath: _hotwordFilePath, // 热词文件可选，传空
            threadsNum: _threadsNum
        );

        initAction.Should().Throw<Exception>()
            .WithMessage("*tokens invalid*"); // 异常信息应包含关键文件标识
    }

    /// <summary>
    /// 测试3：创建 OfflineStream 并添加采样 → 流应正常工作
    /// </summary>
    [Fact]
    public void OfflineRecognizer_CreateStream_AddSamples_ShouldWork()
    {
        // Arrange：先初始化合法识别器，再创建流
        InitRecognizer();
        var mockAudioSamples = GenerateMockAudioSamples(); // 生成模拟音频采样（16kHz，1秒静音）

        // Act：创建流并添加采样
        var stream = _recognizer.CreateOfflineStream();
        stream.AddSamples(mockAudioSamples);

        // Assert：流非空，采样添加无异常（验证接口调用成功）
        stream.Should().NotBeNull();
    }

    /// <summary>
    /// 测试4：获取识别结果 → 结果实体应符合格式（非空，Text/Tokens有默认值）
    /// </summary>
    [Fact]
    public void OfflineRecognizer_GetResult_WithValidStream_ShouldReturnResultEntity()
    {
        // Arrange：初始化识别器、创建流、添加模拟采样
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        var mockSamples = GenerateMockAudioSamples();
        stream.AddSamples(mockSamples);

        // Act：获取识别结果
        var result = _recognizer.GetResult(stream);

        // Assert：结果实体非空，关键字段格式合法（即使是静音，结果也应符合结构）
        result.Should().NotBeNull();
        result.Should().BeOfType<OfflineRecognizerResultEntity>();
        result.Text.Should().NotBeNull(); // 允许空字符串，但不应为 null
        result.Tokens.Should().NotBeNull(); // Tokens 列表不应为 null（可能为空列表）
        result.Timestamps.Should().NotBeNull(); // 时间戳列表不应为 null
    }

    /// <summary>
    /// 辅助方法：生成模拟音频采样（16kHz 采样率，1秒静音，单声道 float 数组）
    /// </summary>
    private float[] GenerateMockAudioSamples(int durationSeconds = 1, int sampleRate = 16000)
    {
        var sampleCount = durationSeconds * sampleRate;
        var samples = new float[sampleCount];
        // 静音采样：所有值为 0（符合音频格式要求）
        Array.Fill(samples, 0.0f);
        return samples;
    }

    /// <summary>
    /// 测试1：添加合法采样（非空 float 数组）→ 无异常
    /// </summary>
    [Fact]
    public void OfflineStream_AddSamples_WithValidSamples_ShouldNotThrow()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange：生成合法采样（1000个 float 元素）
        var validSamples = new float[1000];
        Array.Fill(validSamples, 0.1f); // 非静音采样（模拟真实音频）

        // Act：添加采样（无异常即通过）
        Action addAction = () => stream.AddSamples(validSamples);

        // Assert：无异常抛出
        addAction.Should().NotThrow();
    }

    /// <summary>
    /// 测试2：添加空采样数组 → 应抛出 ArgumentNullException
    /// </summary>
    [Fact]
    public void OfflineStream_AddSamples_WithNullSamples_ShouldThrowArgumentNullException()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange：空采样数组
        float[]? nullSamples = null;

        // Act + Assert：添加空采样 → 抛出空参数异常
        Action addAction = () => stream.AddSamples(nullSamples!);
        addAction.Should().Throw<ArgumentNullException>()
            .WithParameterName("source"); // 异常应指向 "samples" 参数
    }

    /// <summary>
    /// 测试3：设置热词（合法字符串）→ 无异常（热词生效与否不依赖真实识别）
    /// </summary>
    [Fact]
    public void OfflineStream_SetHotwords_WithValidText_ShouldNotThrow()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange：合法热词（如“魔搭”“语音识别”）
        var hotwords = Utils.TextHelper.GetHotwords(Path.Combine(_ModelDir, "tokens.txt"), new string[] { "魔搭", "语音识别", "人工智能" });

        // Act：设置热词（核心库若支持 Hotwords 属性，直接赋值）
        Action setHotwordAction = () => stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(_ModelDir, "tokens.txt"), new string[] { "魔搭", "语音识别", "人工智能" });

        // Assert：无异常抛出（热词格式合法）
        setHotwordAction.Should().NotThrow();
        stream.Hotwords.Should().BeEquivalentTo(hotwords); // 验证热词已设置
    }

    /// <summary>
    /// 测试4：设置空热词 → 应允许（清空热词）
    /// </summary>
    [Fact]
    public void OfflineStream_SetHotwords_WithEmptyText_ShouldClearHotwords()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange：先设置非空热词，再设为空
        stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(_ModelDir, "tokens.txt"), new string[] { "魔搭" });
        var emptyHotwords = "";

        // Act：设置空热词
        stream.Hotwords = null;

        // Assert：热词已清空
        stream.Hotwords.Should().BeNull();
    }

    /// <summary>
    /// 测试5：Dispose 后 → 识别器资源应释放（再次调用方法应抛出异常）
    /// </summary>
    [Fact]
    public void OfflineRecognizer_Dispose_ShouldReleaseResources()
    {
        // Arrange：识别器释放
        InitRecognizer();
        _recognizer.Should().NotBeNull("识别器初始化失败，无法执行测试");
        _recognizer.Dispose();

        // Act + Assert：释放后创建流 → 应抛出异常（资源已释放）
        Action createStreamAfterDispose = () => _recognizer.CreateOfflineStream();
        createStreamAfterDispose
        .Should().Throw<ObjectDisposedException>("释放后的识别器不应再允许创建流")
        .And.ObjectName.Should().Be("OfflineRecognizer", "异常应明确标识被释放的对象名称");
    }

    public void InitRecognizer()
    {
        if (_recognizer == null)
        {
            _recognizer = new OfflineRecognizer(
                modelFilePath: _modelFilePath,
                configFilePath: _configFilePath,
                mvnFilePath: _mvnFilePath,
                tokensFilePath: _tokensFilePath,
                modelebFilePath: _modelebFilePath,
                hotwordFilePath: _hotwordFilePath, // 热词文件可选，传空
                threadsNum: _threadsNum
            );
        }
    }

    /// <summary>
    /// 测试清理：确保识别器资源释放
    /// </summary>
    public void Dispose()
    {
        _recognizer?.Dispose();
    }
}
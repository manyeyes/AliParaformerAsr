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
/// ���� OfflineRecognizer ���Ĺ��ܣ���ʼ��������������ȡ������ͷ���Դ
/// </summary>
public class OfflineRecognizerTests : IDisposable
{
    // ������Դ·����ģ��ģ���ļ�����Ŀ¼
    private string _ModelDir;
    // ģ��ģ���ļ�·����Ԥ�����Ϸ�·����
    private string _modelFilePath;
    private string _configFilePath;
    private string _mvnFilePath;
    private string _tokensFilePath;
    private string _modelebFilePath;
    private string _hotwordFilePath;
    private int _threadsNum = 2;

    // �����Ե�ʶ����ʵ��
    private OfflineRecognizer? _recognizer;

    public OfflineRecognizerTests()
    {
        string modelBasePath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, "TestResources");
        string modelName = "sensevoice-small-int8-onnx";
        // 1. ��ʼ��������Դ·������ȡ TestResources/MockModels Ŀ¼��
        Directory.CreateDirectory(modelBasePath);
        _ModelDir = Path.Combine(
            modelBasePath, modelName
        );

        //// 2.����ģ��
        //GitHelper gitHelper = new GitHelper();
        //Task.Run(() => gitHelper.ProcessCloneModel(modelBasePath, modelName));

        // 3. ����ģ��ģ���ļ�������·����ȷ���ļ����ڣ�
        _modelFilePath = Path.Combine(_ModelDir, "model.int8.onnx");
        _configFilePath = Path.Combine(_ModelDir, "asr.yaml");
        _mvnFilePath = Path.Combine(_ModelDir, "am.mvn");
        _tokensFilePath = Path.Combine(_ModelDir, "tokens.txt");
        _modelebFilePath = Path.Combine(_ModelDir, "model_eb.int8.onnx");

        // 4. ȷ��ģ���ļ����ڣ����������򴴽����ļ�������·������
        EnsureMockFilesExist();
    }

    /// <summary>
    /// ȷ������ģ��ģ���ļ����ڣ����� FileNotFoundException��
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

    //// �����첽����������ȷ��ģ�Ϳ�¡����Ҷ����ʼ����ȷ
    //[Fact]
    //public async Task CreateAsync_ShouldCloneModelAndInitializePath()
    //{
    //    // Arrange��׼������·���Ͳ���
    //    string modelBasePath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, "TestResources");
    //    string modelName = "sensevoice-small-int8-onnx";
    //    string expectedModelPath = Path.Combine(modelBasePath, modelName);

    //    // ������ܵĲ����ļ���ȷ�����Զ����ԣ�
    //    //if (Directory.Exists(expectedModelPath))
    //    //    Directory.Delete(expectedModelPath, recursive: true);

    //    // 2.����ģ��
    //    GitHelper gitHelper = new GitHelper();
    //    await Task.Run(() => gitHelper.ProcessCloneModel(modelBasePath, modelName));

    //    // Assert����֤���
    //    // 1. ģ��Ŀ¼Ӧ����������¡�ɹ���
    //    Assert.True(Directory.Exists(expectedModelPath), "ģ�Ϳ�¡ʧ�ܣ�Ŀ¼δ����");
    //}

    /// <summary>
    /// ����1���Ϸ�������ʼ�� OfflineRecognizer �� Ӧ�ɹ���ʶ�����ǿգ�
    /// </summary>
    [Fact]
    public void OfflineRecognizer_Init_WithValidParams_ShouldReturnNonNull()
    {
        //if (string.IsNullOrEmpty(_modelFilePath) || string.IsNullOrEmpty(_tokensFilePath))
        //{
        //    return;
        //}

        // Act������ʶ����ʵ��
        InitRecognizer();

        // Assert��ʶ�����ǿգ���ʼ���ɹ�
        _recognizer.Should().NotBeNull();
    }

    /// <summary>
    /// ����2��ȱʧ�ؼ��ļ����� tokens.txt���� ��ʼ��Ӧ�׳��쳣
    /// </summary>
    [Fact]
    public void OfflineRecognizer_Init_WithMissingTokensFile_ShouldThrowException()
    {
        // Arrange��ɾ�� tokens.txt��ģ���ļ�ȱʧ��
        //if (File.Exists(_mockTokensPath))
        //{
        //    File.Delete(_mockTokensPath);
        //}

        _tokensFilePath = "";

        // Act + Assert����ʼ��Ӧ�׳��쳣���ؼ��ļ�ȱʧ��
        Action initAction = () => new OfflineRecognizer(
            modelFilePath: _modelFilePath,
            configFilePath: _configFilePath,
            mvnFilePath: _mvnFilePath,
            tokensFilePath: _tokensFilePath,
            modelebFilePath: _modelebFilePath,
            hotwordFilePath: _hotwordFilePath, // �ȴ��ļ���ѡ������
            threadsNum: _threadsNum
        );

        initAction.Should().Throw<Exception>()
            .WithMessage("*tokens invalid*"); // �쳣��ϢӦ�����ؼ��ļ���ʶ
    }

    /// <summary>
    /// ����3������ OfflineStream ����Ӳ��� �� ��Ӧ��������
    /// </summary>
    [Fact]
    public void OfflineRecognizer_CreateStream_AddSamples_ShouldWork()
    {
        // Arrange���ȳ�ʼ���Ϸ�ʶ�������ٴ�����
        InitRecognizer();
        var mockAudioSamples = GenerateMockAudioSamples(); // ����ģ����Ƶ������16kHz��1�뾲����

        // Act������������Ӳ���
        var stream = _recognizer.CreateOfflineStream();
        stream.AddSamples(mockAudioSamples);

        // Assert�����ǿգ�����������쳣����֤�ӿڵ��óɹ���
        stream.Should().NotBeNull();
    }

    /// <summary>
    /// ����4����ȡʶ���� �� ���ʵ��Ӧ���ϸ�ʽ���ǿգ�Text/Tokens��Ĭ��ֵ��
    /// </summary>
    [Fact]
    public void OfflineRecognizer_GetResult_WithValidStream_ShouldReturnResultEntity()
    {
        // Arrange����ʼ��ʶ�����������������ģ�����
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        var mockSamples = GenerateMockAudioSamples();
        stream.AddSamples(mockSamples);

        // Act����ȡʶ����
        var result = _recognizer.GetResult(stream);

        // Assert�����ʵ��ǿգ��ؼ��ֶθ�ʽ�Ϸ�����ʹ�Ǿ��������ҲӦ���Ͻṹ��
        result.Should().NotBeNull();
        result.Should().BeOfType<OfflineRecognizerResultEntity>();
        result.Text.Should().NotBeNull(); // ������ַ���������ӦΪ null
        result.Tokens.Should().NotBeNull(); // Tokens �б�ӦΪ null������Ϊ���б�
        result.Timestamps.Should().NotBeNull(); // ʱ����б�ӦΪ null
    }

    /// <summary>
    /// ��������������ģ����Ƶ������16kHz �����ʣ�1�뾲���������� float ���飩
    /// </summary>
    private float[] GenerateMockAudioSamples(int durationSeconds = 1, int sampleRate = 16000)
    {
        var sampleCount = durationSeconds * sampleRate;
        var samples = new float[sampleCount];
        // ��������������ֵΪ 0��������Ƶ��ʽҪ��
        Array.Fill(samples, 0.0f);
        return samples;
    }

    /// <summary>
    /// ����1����ӺϷ��������ǿ� float ���飩�� ���쳣
    /// </summary>
    [Fact]
    public void OfflineStream_AddSamples_WithValidSamples_ShouldNotThrow()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange�����ɺϷ�������1000�� float Ԫ�أ�
        var validSamples = new float[1000];
        Array.Fill(validSamples, 0.1f); // �Ǿ���������ģ����ʵ��Ƶ��

        // Act����Ӳ��������쳣��ͨ����
        Action addAction = () => stream.AddSamples(validSamples);

        // Assert�����쳣�׳�
        addAction.Should().NotThrow();
    }

    /// <summary>
    /// ����2����ӿղ������� �� Ӧ�׳� ArgumentNullException
    /// </summary>
    [Fact]
    public void OfflineStream_AddSamples_WithNullSamples_ShouldThrowArgumentNullException()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange���ղ�������
        float[]? nullSamples = null;

        // Act + Assert����ӿղ��� �� �׳��ղ����쳣
        Action addAction = () => stream.AddSamples(nullSamples!);
        addAction.Should().Throw<ArgumentNullException>()
            .WithParameterName("source"); // �쳣Ӧָ�� "samples" ����
    }

    /// <summary>
    /// ����3�������ȴʣ��Ϸ��ַ������� ���쳣���ȴ���Ч���������ʵʶ��
    /// </summary>
    [Fact]
    public void OfflineStream_SetHotwords_WithValidText_ShouldNotThrow()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange���Ϸ��ȴʣ��硰ħ�������ʶ�𡱣�
        var hotwords = Utils.TextHelper.GetHotwords(Path.Combine(_ModelDir, "tokens.txt"), new string[] { "ħ��", "����ʶ��", "�˹�����" });

        // Act�������ȴʣ����Ŀ���֧�� Hotwords ���ԣ�ֱ�Ӹ�ֵ��
        Action setHotwordAction = () => stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(_ModelDir, "tokens.txt"), new string[] { "ħ��", "����ʶ��", "�˹�����" });

        // Assert�����쳣�׳����ȴʸ�ʽ�Ϸ���
        setHotwordAction.Should().NotThrow();
        stream.Hotwords.Should().BeEquivalentTo(hotwords); // ��֤�ȴ�������
    }

    /// <summary>
    /// ����4�����ÿ��ȴ� �� Ӧ��������ȴʣ�
    /// </summary>
    [Fact]
    public void OfflineStream_SetHotwords_WithEmptyText_ShouldClearHotwords()
    {
        InitRecognizer();
        var stream = _recognizer.CreateOfflineStream();
        // Arrange�������÷ǿ��ȴʣ�����Ϊ��
        stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(_ModelDir, "tokens.txt"), new string[] { "ħ��" });
        var emptyHotwords = "";

        // Act�����ÿ��ȴ�
        stream.Hotwords = null;

        // Assert���ȴ������
        stream.Hotwords.Should().BeNull();
    }

    /// <summary>
    /// ����5��Dispose �� �� ʶ������ԴӦ�ͷţ��ٴε��÷���Ӧ�׳��쳣��
    /// </summary>
    [Fact]
    public void OfflineRecognizer_Dispose_ShouldReleaseResources()
    {
        // Arrange��ʶ�����ͷ�
        InitRecognizer();
        _recognizer.Should().NotBeNull("ʶ������ʼ��ʧ�ܣ��޷�ִ�в���");
        _recognizer.Dispose();

        // Act + Assert���ͷź󴴽��� �� Ӧ�׳��쳣����Դ���ͷţ�
        Action createStreamAfterDispose = () => _recognizer.CreateOfflineStream();
        createStreamAfterDispose
        .Should().Throw<ObjectDisposedException>("�ͷź��ʶ������Ӧ����������")
        .And.ObjectName.Should().Be("OfflineRecognizer", "�쳣Ӧ��ȷ��ʶ���ͷŵĶ�������");
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
                hotwordFilePath: _hotwordFilePath, // �ȴ��ļ���ѡ������
                threadsNum: _threadsNum
            );
        }
    }

    /// <summary>
    /// ��������ȷ��ʶ������Դ�ͷ�
    /// </summary>
    public void Dispose()
    {
        _recognizer?.Dispose();
    }
}
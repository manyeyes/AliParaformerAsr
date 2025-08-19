using MauiApp1.Utils;
using System.Text;

namespace MauiApp1;

public partial class RecognitionForFiles : ContentPage
{
    private string _rootFolderName = "AllModels";
    private string[] _downloadHostParams = new string[] { "https://www.modelscope.cn/models", "master" };
    //private string _subFolderName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
    private string _subFolderName = "sensevoice-small-onnx";
    private Dictionary<string, string> _modelFiles = new Dictionary<string, string>() {
        { "model.int8.onnx","d2164f971d7c936d2bf3cbe9d5f43d1e"},
        {"am.mvn","dc1dbdeeb8961f012161cfce31eaacaf" },
        {"asr.json","56ea6051f22a1b3f3e7506a963123a74" },
        {"tokens.txt","56b7ae79411b22f167a1f185cae94aa7" },
        {"0.wav","af1295e53df7298458f808bc0cd946e2" }
    };
    private AliParaformerAsr.OfflineRecognizer _offlineRecognizer = null;

    public RecognitionForFiles()
    {
        InitializeComponent();
        //GetDownloadState();
    }

    private async void OnDownLoadCheckClicked(object sender, EventArgs e)
    {
        BtnDownLoadCheck.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            DownloadCheck();
        });
        BtnDownLoadCheck.IsEnabled = true;
    }

    private async void OnDownLoadModelsClicked(object sender, EventArgs e)
    {
        BtnDownLoadModels.IsEnabled = false;
        DownloadProgressBar.Progress = 0 / 100.0;
        DownloadProgressLabel.Text = "";
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            DownloadModels();
        });
        BtnDownLoadModels.IsEnabled = true;
    }

    async void DownloadModels()
    {
        List<string> indexs = new List<string>();
        DownloadHelper downloadHelper = new DownloadHelper(this.DownloadDisplay);
        DownloadProgressLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressLabel.IsVisible = true;
                                 DownloadProgressLabel.Text = "";
                             }));

        DownloadProgressBar.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressBar.IsVisible = true;
                                 DownloadProgressBar.Progress = 0 / 100.0;
                             }));
        while (indexs.Count < _modelFiles.Count)
        {
            if (downloadHelper.IsDownloading)
            {
                continue;
            }
            foreach (var modelFile in _modelFiles)
            {
                var fileName = modelFile.Key;
                if (indexs.Contains(fileName))
                {
                    continue;
                }
                if (downloadHelper.IsDownloading)
                {
                    break;
                }
                var downloadUrl = string.Format("{0}/manyeyes/{1}/resolve/{2}/{3}", _downloadHostParams[0], _subFolderName, _downloadHostParams[1], fileName);
                downloadHelper.DownloadCreate(downloadUrl, fileName, _rootFolderName, _subFolderName);
                downloadHelper.DownloadStart();
                indexs.Add(fileName);
            }
        }
        DownloadProgressLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressLabel.IsVisible = false;
                                 DownloadProgressLabel.Text = "";
                             }));

        DownloadProgressBar.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressBar.IsVisible = false;
                                 DownloadProgressBar.Progress = 0 / 100.0;
                             }));
    }

    async void GetDownloadState()
    {
        DownloadHelper downloadHelper = new DownloadHelper();
        bool state = downloadHelper.GetDownloadState(_modelFiles, _rootFolderName, _subFolderName);
        ModelStatusLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 if (state)
                                 {
                                     ModelStatusLabel.Text = "model is ready";
                                 }
                                 else
                                 {
                                     ModelStatusLabel.Text = "model not ready";
                                 }

                             }));
    }

    async void DownloadCheck()
    {

        DownloadHelper downloadHelper = new DownloadHelper(this.DownloadDisplay);
        bool state = downloadHelper.GetDownloadState(_modelFiles, _rootFolderName, _subFolderName);
        ModelStatusLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 if (state)
                                 {
                                     ModelStatusLabel.Text = "model is ready";
                                 }
                                 else
                                 {
                                     ModelStatusLabel.Text = "model not ready";
                                 }

                             }));
        if (!state)
        {
            DownloadResultsLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadResultsLabel.IsVisible = true;
                                 DownloadResultsLabel.Text = "";
                             }));
            foreach (var modelFile in _modelFiles)
            {
                downloadHelper.DownloadCheck(modelFile.Key, _rootFolderName, _subFolderName, modelFile.Value);
            }
        }
    }


    private void DownloadDisplay(int progress, DownloadState downloadState, string filename, string msg = "")
    {

        switch (downloadState)
        {
            case DownloadState.inprogres:
                DownloadProgressBar.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressBar.Progress = progress / 100.0;
                             }));
                DownloadProgressLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressLabel.Text = $"文件：{filename}，正在下载，进度：{progress}%\n";
                             }));

                break;
            case DownloadState.cancelled:
                DownloadProgressBar.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressBar.Progress = progress / 100.0;
                             }));
                DownloadProgressLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressLabel.Text = $"文件：{filename}，下载已取消\n";
                             }));
                break;
            case DownloadState.error:
                DownloadProgressBar.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressBar.Progress = progress / 100.0;
                             }));
                DownloadProgressLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressLabel.Text = $"文件：{filename}，下载失败：{msg}\n";
                             }));
                DownloadResultsLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadResultsLabel.Text += $"文件：{filename}，下载失败：{msg}\n";
                             }));
                break;
            case DownloadState.completed:
                DownloadProgressBar.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressBar.Progress = progress / 100.0;
                             }));
                DownloadProgressLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressLabel.Text = $"文件：{filename}，下载完成\n";
                             }));
                DownloadResultsLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadResultsLabel.Text += $"文件：{filename}，下载完成\n";
                             }));
                break;
            case DownloadState.existed:
                DownloadProgressBar.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadProgressBar.Progress = progress / 100.0;
                             }));
                DownloadResultsLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadResultsLabel.Text += $"文件：{filename}，已存在\n";
                             }));
                break;
            case DownloadState.noexisted:
                DownloadResultsLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadResultsLabel.Text += $"文件：{filename}，不存在\n";
                             }));
                break;
        }
    }

    private async void OnBtnRecognitionExampleClicked(object sender, EventArgs e)
    {
        BtnRecognitionExample.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            TestAliParaformerAsrOfflineRecognizer();
        });
        BtnRecognitionExample.IsEnabled = true;
    }

    private async void OnBtnRecognitionFilesClicked(object sender, EventArgs e)
    {
        var customFileType = new FilePickerFileType(
                new Dictionary<DevicePlatform, IEnumerable<string>>
                {
                    { DevicePlatform.iOS, new[] { "public.my.comic.extension" } }, // UTType values
                    { DevicePlatform.Android, new[] { "audio/x-wav" } }, // MIME type
                    { DevicePlatform.WinUI, new[] { ".wav", ".mp3" } }, // file extension
                    { DevicePlatform.Tizen, new[] { "*/*" } },
                    { DevicePlatform.macOS, new[] { "cbr", "cbz" } }, // UTType values
                });

        PickOptions options = new()
        {
            PickerTitle = "Please select a comic file",
            FileTypes = customFileType,
        };


        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            var fileResult = await PickAndShow(options);
            if (fileResult != null)
            {
                string fullpath = fileResult.FullPath;
                List<string> fullpaths = new List<string>();
                fullpaths.Add(fullpath);
                RecognizerFilesByAliParaformerAsrOffline(fullpaths);
            }
        });
    }

    public async Task<FileResult> PickAndShow(PickOptions options)
    {
        try
        {
            var result = await FilePicker.Default.PickAsync(options);
            return result;
        }
        catch (Exception ex)
        {
            // The user canceled or something went wrong
        }

        return null;
    }

    public void CreateDownloadFile(string fileName)
    {

        var downloadFolder = FileSystem.AppDataDirectory + "/Download/";
        Directory.CreateDirectory(downloadFolder);
        var filePath = downloadFolder + fileName;
        File.Create(filePath);
    }

    public AliParaformerAsr.OfflineRecognizer initAliParaformerAsrOfflineRecognizer(string modelName)
    {
        if (_offlineRecognizer == null)
        {
            try
            {
                string allModels = "AllModels";
                string modelFilePath = SysConf.ApplicationBase + "/" + allModels + "/" + modelName + "/model.int8.onnx";
                string configFilePath = SysConf.ApplicationBase + "/" + allModels + "/" + modelName + "/asr.json";
                string mvnFilePath = SysConf.ApplicationBase + "/" + allModels + "/" + modelName + "/am.mvn";
                string tokensFilePath = SysConf.ApplicationBase + "/" + allModels + "/" + modelName + "/tokens.txt";
                _offlineRecognizer = new AliParaformerAsr.OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath, threadsNum: 5);
            }
            catch
            {
                DisplayAlert("Tips", "Please check if the model is correct", "close");
            }
        }
        return _offlineRecognizer;
    }

    public void RecognizerFilesByAliParaformerAsrOffline(List<string> fullpaths)
    {
        AliParaformerAsr.OfflineRecognizer offlineRecognizer = initAliParaformerAsrOfflineRecognizer(_subFolderName);
        if (offlineRecognizer == null) { return; }
        AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "Speech recognition in progress, please wait……";
                             }
                             ));
        List<float[]>? samples = new List<float[]>();
        List<string> paths = new List<string>();
        TimeSpan total_duration = new TimeSpan(0L);
        foreach (string fullpath in fullpaths)
        {
            string mediaFilePath = string.Format(fullpath);
            if (!File.Exists(mediaFilePath))
            {
                continue;
            }
            if (AudioHelper.IsAudioByHeader(mediaFilePath))
            {
                TimeSpan duration = TimeSpan.Zero;
                float[]? sample = AudioHelper.GetFileSample(wavFilePath: mediaFilePath, duration: ref duration);
                if (sample != null)
                {
                    paths.Add(mediaFilePath);
                    samples.Add(sample);
                    total_duration += duration;
                }
            }
        }
        if (samples.Count == 0)
        {
            AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "No media file is read!";
                             }
                             ));
            return;
        }
        AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                                 List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
                                 foreach (var sample in samples)
                                 {
                                     AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                                     stream.AddSamples(sample);
                                     streams.Add(stream);
                                 }
                                 List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
                                 int n = 0;
                                 foreach (AliParaformerAsr.Model.OfflineRecognizerResultEntity result in results)
                                 {
                                     StringBuilder r = new StringBuilder();
                                     r.Append("{");
                                     r.Append($"\"path\": \"{paths[n]}\",");
                                     r.Append($"\"text\": \"{result?.Text}\",");
                                     r.Append($"\"tokens\":[{string.Join(",", result?.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                                     r.Append($"\"timestamps\":[{string.Join(",", result?.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                                     r.Append("}\n");
                                     AsrResults.Text = r.ToString();
                                     n++;
                                 }
                                 TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                                 double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                                 double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
                                 AsrResults.Text += string.Format("elapsed_milliseconds:{0}\n", elapsed_milliseconds.ToString());
                                 AsrResults.Text += string.Format("total_duration:{0}\n", total_duration.TotalMilliseconds.ToString());
                                 AsrResults.Text += string.Format("rtf:{1}\n", "0".ToString(), rtf.ToString());
                                 AsrResults.Text += string.Format("End!");
                             }));

    }

    public void TestAliParaformerAsrOfflineRecognizer(List<float[]>? samples = null)
    {
        AliParaformerAsr.OfflineRecognizer offlineRecognizer = initAliParaformerAsrOfflineRecognizer(_subFolderName);
        if (offlineRecognizer == null) { return; }
        AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "Speech recognition in progress, please wait ...";
                             }
                             ));
        TimeSpan total_duration = new TimeSpan(0L);
        List<string> paths = new List<string>();
        try
        {
            if (samples == null)
            {
                samples = new List<float[]>();
                string[]? mediaFilePaths = null;
                if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
                {
                    string fullPath = Path.Combine(SysConf.ApplicationBase + "/AllModels/", _subFolderName);
                    if (!Directory.Exists(fullPath))
                    {
                        mediaFilePaths = Array.Empty<string>(); // 路径不正确时返回空数组
                    }
                    else
                    {
                        //mediaFilePaths = Directory.GetFiles(fullPath);
                        mediaFilePaths = Directory.GetFiles(
                            path: fullPath,
                            searchPattern: "*.wav",
                            searchOption: SearchOption.AllDirectories
                        );
                    }
                }
                foreach (string mediaFilePath in mediaFilePaths)
                {
                    if (!File.Exists(mediaFilePath))
                    {
                        continue;
                    }
                    if (AudioHelper.IsAudioByHeader(mediaFilePath))
                    {
                        TimeSpan duration = TimeSpan.Zero;
                        float[]? sample = AudioHelper.GetFileSample(wavFilePath: mediaFilePath, duration: ref duration);
                        if (sample != null)
                        {
                            paths.Add(mediaFilePath);
                            samples.Add(sample);
                            total_duration += duration;
                        }
                    }
                }
            }
            if (samples.Count == 0)
            {                
                AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "No media file is read!";
                             }
                             ));
                return;
            }
            AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                                     List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
                                     foreach (var sample in samples)
                                     {
                                         AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                                         stream.AddSamples(sample);
                                         streams.Add(stream);
                                     }
                                     List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
                                     int n = 0;
                                     foreach (AliParaformerAsr.Model.OfflineRecognizerResultEntity result in results)
                                     {
                                         StringBuilder r = new StringBuilder();
                                         r.Append("{");
                                         r.Append($"\"path\": \"{paths[n]}\",");
                                         r.Append($"\"text\": \"{result?.Text}\",");
                                         r.Append($"\"tokens\":[{string.Join(",", result?.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                                         r.Append($"\"timestamps\":[{string.Join(",", result?.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                                         r.Append("}\n");
                                         AsrResults.Text = r.ToString();
                                         n++;
                                     }
                                     TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                                     double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                                     double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
                                     Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
                                     Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
                                     Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
                                     Console.WriteLine("Hello, World!");
                                     AsrResults.Text += string.Format("elapsed_milliseconds:{0}\n", elapsed_milliseconds.ToString());
                                     AsrResults.Text += string.Format("total_duration:{0}\n", total_duration.TotalMilliseconds.ToString());
                                     AsrResults.Text += string.Format("rtf:{1}\n", "0".ToString(), rtf.ToString());
                                     AsrResults.Text += string.Format("End!");
                                 }
                                 ));
        }
        catch(Exception ex)
        {
            AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     AsrResults.Text = ex.Message;
                                 }
                                 ));
        }

    }
}


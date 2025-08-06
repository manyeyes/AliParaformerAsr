// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
/*
 * Before running, please prepare the model first
 * Model Download:
 * Please read README.md
 */
namespace AliParaformerAsr.Examples;
internal static partial class Program
{
    public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
    [STAThread]
    private static void Main()
    {
        string defaultOfflineModelName = "aliparaformerasr-large-zh-en-timestamp-onnx-offline";
        OfflineRecognizer(streamDecodeMethod: "one", modelName: defaultOfflineModelName);
        string defaultOnlineModelName = "aliparaformerasr-large-zh-en-onnx-online";
        OnlineRecognizer(streamDecodeMethod: "one", modelName: defaultOnlineModelName);
        //suggest GC recycling (non mandatory)
        GC.Collect(); //trigger recycling
        GC.WaitForPendingFinalizers(); //waiting for the terminator to complete execution
        GC.Collect(); //recycling again
    }
}
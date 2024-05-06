using AliParaformerAsr;
using NAudio.Wave;

namespace AliParaformerAsr.Examples;
internal static partial class Program
{
    public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
    [STAThread]
    private static void Main()
    {
        OfflineRecognizer();
        OnlineRecognizer();
    }
}
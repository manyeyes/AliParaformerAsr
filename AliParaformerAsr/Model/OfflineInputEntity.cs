// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace AliParaformerAsr.Model
{
    public class OfflineInputEntity
    {
        public float[]? Speech { get; set; }
        public int SpeechLength { get; set; }
        public List<int[]>? Hotwords { get; set; } = new List<int[]>();
    }
}

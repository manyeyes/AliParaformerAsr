// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace AliParaformerAsr.Model
{
    public class OfflineInputEntity
    {
        private float[]? _speech;
        private int _speech_length=0;

        //public List<float[]>? speech { get; set; }
        public float[]? Speech { get; set; }
        public int SpeechLength { get; set; }
    }
}

﻿// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace AliParaformerAsr.Model
{
    /// <summary>
    /// online recognizer result entity 
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OfflineRecognizerResultEntity
    {
        private List<string>? _tokens=new List<string>();
        private List<int[]>? _timestamps=new List<int[]>();
        /// <summary>
        /// recognizer result
        /// </summary>
        public string? Text { get; set; }
        /// <summary>
        /// recognizer result length
        /// </summary>
        public int TextLen { get; set; }
        /// <summary>
        /// decode tokens
        /// </summary>
        //public List<string>? tokens { get; set; }
        public List<string>? Tokens { get => _tokens; set => _tokens = value; }

        /// <summary>
        /// timestamps
        /// </summary>
        //public List<int[]>? timestamps { get; set; }
        public List<int[]>? Timestamps { get => _timestamps; set => _timestamps = value; }

    }
}

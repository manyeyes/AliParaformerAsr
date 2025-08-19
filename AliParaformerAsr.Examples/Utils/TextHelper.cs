using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliParaformerAsr.Examples.Utils
{
    internal class TextHelper
    {
        public static List<int[]>? GetHotwords(string tokensFilePath, string hotwordFilePath)
        {
            List<int[]>? hotwords = new List<int[]>();
            if (File.Exists(tokensFilePath) && File.Exists(hotwordFilePath))
            {
                string[] tokens = File.ReadAllLines(tokensFilePath);
                string[] sentences = File.ReadAllLines(hotwordFilePath);
                foreach (string sentence in sentences)
                {
                    string[] wordList = new string[] { sentence };//TODO:分词
                    foreach (string word in wordList)
                    {
                        List<int> ids = word.ToCharArray().Select(x => Array.IndexOf(tokens, x.ToString())).Where(x => x != -1).ToList();
                        hotwords.Add(ids.ToArray());
                    }
                }
                hotwords.Add(new int[] { 1 });
            }
            return hotwords;
        }
        public static List<int[]>? GetHotwords(string tokensFilePath, string[] wordList)
        {
            List<int[]>? hotwords = new List<int[]>();
            if (File.Exists(tokensFilePath) && wordList.Length>0)
            {
                string[] tokens = File.ReadAllLines(tokensFilePath);
                foreach (string word in wordList)
                {
                    List<int> ids = word.ToCharArray().Select(x => Array.IndexOf(tokens, x.ToString())).Where(x => x != -1).ToList();
                    hotwords.Add(ids.ToArray());
                }
                hotwords.Add(new int[] { 1 });
            }
            return hotwords;
        }
    }
}

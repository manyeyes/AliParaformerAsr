// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using System.IO;
using System.Reflection;
using System.Text.Json;
using YamlDotNet.Core.Tokens;
using YamlDotNet.Serialization;

namespace AliParaformerAsr.Utils
{
    /// <summary>
    /// PreloadHelper
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    internal class PreloadHelper
    {
        public static T? ReadYaml<T>(string yamlFilePath)
        {
            T? info = default(T);
            if (!string.IsNullOrEmpty(yamlFilePath) && yamlFilePath.IndexOf("/") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(yamlFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{yamlFilePath}' not found.");
                using (var yamlReader = new StreamReader(stream))
                {
                    Deserializer yamlDeserializer = new Deserializer();
                    info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
            else if (File.Exists(yamlFilePath))
            {
                using (var yamlReader = File.OpenText(yamlFilePath))
                {
                    Deserializer yamlDeserializer = new Deserializer();
                    info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
#pragma warning disable CS8603 // 可能返回 null 引用。
            return info;
#pragma warning restore CS8603 // 可能返回 null 引用。
        }

        public static T? ReadJson<T>(string jsonFilePath)
        {
            T? info = default(T);
            if (!string.IsNullOrEmpty(jsonFilePath) && jsonFilePath.IndexOf("/") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(jsonFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{jsonFilePath}' not found.");
                using (var jsonReader = new StreamReader(stream))
                {
                    Deserializer jsonDeserializer = new Deserializer();
                    info = JsonSerializer.Deserialize<T>(jsonReader.ReadToEnd());
                    jsonReader.Close();
                }
            }
            else if (File.Exists(jsonFilePath))
            {
                using (var jsonReader = File.OpenText(jsonFilePath))
                {
                    info = JsonSerializer.Deserialize<T>(jsonReader.ReadToEnd());
                    jsonReader.Close();
                }
            }
            return info;
        }

        public static string[] ReadTokens(string tokensFilePath)
        {
            string[] tokens = null;
            if (!string.IsNullOrEmpty(tokensFilePath) && tokensFilePath.IndexOf("/") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(tokensFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{tokensFilePath}' not found.");
                using (var reader = new StreamReader(stream))
                {
                    tokens = reader.ReadToEnd().Split("\n");//Environment.NewLine
                }
            }
            else
            {
                tokens = File.ReadAllLines(tokensFilePath);
            }
            return tokens;
        }
    }
}

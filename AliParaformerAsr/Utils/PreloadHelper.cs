// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using System.Reflection;
using YamlDotNet.Serialization;
// 根据框架条件导入命名空间
#if NETCOREAPP3_1_OR_GREATER || NET6_0_OR_GREATER || NETSTANDARD2_0_OR_GREATER || WINDOWS || __ANDROID__ || __IOS__
// 原生支持System.Text.Json的框架：优先使用
using JsonSerializer = System.Text.Json.JsonSerializer;
#else
// 不支持的框架：回退到Newtonsoft.Json
using JsonSerializer = Newtonsoft.Json.JsonConvert;
#endif

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
                    tokens = reader.ReadToEnd().Split('\n');//Environment.NewLine
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

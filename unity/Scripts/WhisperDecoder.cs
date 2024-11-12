using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using System.Text;
using System.Linq;

public class WhisperDecoder : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField] private NNModel modelAsset;
    [SerializeField] private TextAsset characterMapJson;
    
    [Header("Audio Settings")]
    [SerializeField] private int sampleRate = 16000;
    [SerializeField] private int melBands = 80;
    [SerializeField] private int timeFrames = 3000;

    private Model runtimeModel;
    private IWorker worker;
    private Dictionary<string, string> characterMap;

    private void Awake()
    {
        // Load model
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(runtimeModel);

        // Load character map
        LoadCharacterMap();
    }

    private void LoadCharacterMap()
    {
        if (characterMapJson == null)
        {
            Debug.LogError("Character map JSON file not assigned!");
            return;
        }

        try
        {
            characterMap = JsonConvert.DeserializeObject<Dictionary<string, string>>(characterMapJson.text);
            Debug.Log($"Loaded {characterMap.Count} tokens in character map");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error loading character map: {e.Message}");
        }
    }

    public string ProcessAudio(float[] audioData)
    {
        try
        {
            // 1. Convert audio to mel spectrogram
            float[,,] melSpectrogram = ConvertToMelSpectrogram(audioData);

            // 2. Create input tensor
            using (var input = new Tensor(1, melBands, timeFrames, 1, melSpectrogram))
            {
                // 3. Run inference
                var output = worker.Execute(input).PeekOutput();

                // 4. Process output and decode tokens
                return DecodeOutput(output);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error processing audio: {e.Message}");
            return string.Empty;
        }
    }

    private float[,,] ConvertToMelSpectrogram(float[] audioData)
    {
        // Create array for mel spectrogram
        float[,,] melSpec = new float[1, melBands, timeFrames];
        // TODO: Implement mel spectrogram conversion
        return melSpec;
    }

    private string DecodeOutput(Tensor output)
    {
        var logits = output.ToReadOnlyArray();
        var tokens = new List<int>();
        int vocabSize = characterMap.Count;
        
        for (int i = 0; i < output.length; i += vocabSize)
        {
            float maxValue = float.MinValue;
            int maxIndex = 0;
            
            for (int j = 0; j < vocabSize; j++)
            {
                if (logits[i + j] > maxValue)
                {
                    maxValue = logits[i + j];
                    maxIndex = j;
                }
            }
            
            tokens.Add(maxIndex);
        }

        StringBuilder result = new StringBuilder();
        foreach (int token in tokens)
        {
            if (characterMap.TryGetValue(token.ToString(), out string text))
            {
                result.Append(text);
            }
        }

        return result.ToString();
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}

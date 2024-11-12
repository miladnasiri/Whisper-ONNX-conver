# Whisper Character Map Documentation

This document explains the character map structure used for decoding Whisper ONNX model outputs.

## Overview
- Total tokens: 50,363
- Includes special tokens and regular characters
- Used for decoding model outputs to text

## Special Tokens
- `<|startoftranscript|>` (ID: 50258)
- `<|endoftext|>` (ID: 50257)
- `<|transcribe|>` (ID: 50359)
- `<|translate|>` (ID: 50358)
- `<|notimestamps|>` (ID: 50363)

## Structure
The character map is stored in `whisper_char_map.json` with the following structure:
```json
{
    "token_id": "token_text",
    // Example:
    "0": "!",
    "1": "\""
}
```

## Usage in Unity
```csharp
// Example Unity usage
public class WhisperDecoder
{
    private Dictionary<int, string> characterMap;
    
    public void LoadCharacterMap(string jsonPath)
    {
        // Load the JSON file
        string jsonContent = File.ReadAllText(jsonPath);
        characterMap = JsonUtility.FromJson<Dictionary<int, string>>(jsonContent);
    }
    
    public string DecodeTokens(int[] tokens)
    {
        StringBuilder result = new StringBuilder();
        foreach (int token in tokens)
        {
            if (characterMap.TryGetValue(token, out string text))
            {
                result.Append(text);
            }
        }
        return result.ToString();
    }
}
```

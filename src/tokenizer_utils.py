import whisper
from typing import Dict, List

class WhisperTokenizer:
    def __init__(self, model_type: str = "tiny"):
        self.model = whisper.load_model(model_type)
        self.tokenizer = self.model.tokenizer
        self.character_map = self._create_character_map()
        
    def _create_character_map(self) -> Dict[int, str]:
        char_map = {}
        special_tokens = {
            self.tokenizer.transcribe: "<|transcribe|>",
            self.tokenizer.translate: "<|translate|>",
            self.tokenizer.sot: "<|startoftranscript|>",
            self.tokenizer.eot: "<|endoftext|>",
            self.tokenizer.no_timestamps: "<|notimestamps|>",
        }
        
        for token_id, token_text in special_tokens.items():
            char_map[token_id] = token_text
            
        for token_id in range(self.tokenizer.vocab_size):
            if token_id not in char_map:
                char_map[token_id] = self.tokenizer.decode([token_id])
                
        return char_map
    
    def decode_tokens(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def save_char_map(self, filepath: str):
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.character_map, f, ensure_ascii=False, indent=2)

def main():
    tokenizer = WhisperTokenizer("tiny")
    tokenizer.save_char_map("whisper_char_map.json")
    
    text = "Hello, this is a test"
    tokens = tokenizer.encode_text(text)
    decoded = tokenizer.decode_tokens(tokens)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")

if __name__ == "__main__":
    main()

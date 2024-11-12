import whisper
from whisper.tokenizer import get_tokenizer
from typing import Dict, List
import numpy as np

class WhisperTokenizer:
    def __init__(self, model_type: str = "tiny"):
        self.model = whisper.load_model(model_type)
        self.tokenizer = get_tokenizer(multilingual=True)
        self.character_map = self._create_character_map()
        
    def _create_character_map(self) -> Dict[int, str]:
        char_map = {}
        
        # Special tokens
        special_tokens = {
            self.tokenizer.sot: "<|startoftranscript|>",
            self.tokenizer.eot: "<|endoftext|>",
            self.tokenizer.transcribe: "<|transcribe|>",
            self.tokenizer.translate: "<|translate|>",
            self.tokenizer.no_timestamps: "<|notimestamps|>",
        }
        
        # Add special tokens
        for token_id, token_text in special_tokens.items():
            char_map[token_id] = token_text
            
        # Add regular tokens
        for i in range(51865):  # Whisper's vocabulary size
            if i not in char_map:
                decoded = self.tokenizer.decode([i])
                if decoded:
                    char_map[i] = decoded
                
        return char_map
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens to text"""
        return self.tokenizer.decode(tokens)
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to tokens"""
        return self.tokenizer.encode(text)
    
    def save_char_map(self, filepath: str):
        """Save character map to file"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.character_map, f, ensure_ascii=False, indent=2)
    
    def test_tokenizer(self, text: str = "Hello, this is a test."):
        """Test tokenizer functionality"""
        print(f"\nTesting tokenizer with text: {text}")
        tokens = self.encode_text(text)
        decoded = self.decode_tokens(tokens)
        print(f"Encoded tokens: {tokens}")
        print(f"Decoded text: {decoded}")
        return tokens, decoded

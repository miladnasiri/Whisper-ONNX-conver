from src.tokenizer_utils import WhisperTokenizer

def test_tokenizer():
    tokenizer = WhisperTokenizer("tiny")
    tokenizer.save_char_map("whisper_char_map.json")
    
    test_text = "Testing Whisper tokenization"
    tokens = tokenizer.encode_text(test_text)
    decoded = tokenizer.decode_tokens(tokens)
    
    print("\nTest Results:")
    print(f"Original text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded text: {decoded}")

if __name__ == "__main__":
    test_tokenizer()

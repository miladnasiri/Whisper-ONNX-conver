import torch
import whisper
import onnx
import onnxruntime as ort
import numpy as np
import os
from src.tokenizer_utils import WhisperTokenizer

def test_existing_model():
    print("🔍 Testing Existing Whisper ONNX Model")
    
    try:
        # Get the path to tiny.onnx in current directory
        current_dir = os.getcwd()
        onnx_path = os.path.join(current_dir, "tiny.onnx")
        
        print(f"Looking for ONNX model at: {onnx_path}")
        
        # 1. Load and verify ONNX model
        print("\n1. Checking ONNX model...")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model structure is valid")
        
        # 2. Create ONNX Runtime session
        print("\n2. Creating inference session...")
        session = ort.InferenceSession(onnx_path)
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"✓ Input name: {input_name}")
        print(f"✓ Expected input shape: {input_shape}")
        
        # 3. Create and test tokenizer
        print("\n3. Setting up tokenizer...")
        tokenizer = WhisperTokenizer("tiny")
        print("✓ Tokenizer initialized")
        
        # Test tokenizer
        test_text = "Testing Whisper speech recognition."
        tokens, decoded = tokenizer.test_tokenizer(test_text)
        print("✓ Tokenizer test successful")
        
        # 4. Save character map
        print("\n4. Generating character map...")
        char_map_path = os.path.join(current_dir, "whisper_char_map.json")
        tokenizer.save_char_map(char_map_path)
        print(f"✓ Character map saved to {char_map_path}")
        
        # 5. Test model inference
        print("\n5. Testing inference...")
        dummy_input = np.random.randn(1, 80, 3000).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print("✓ Model successfully ran inference")
        print(f"Output shape: {outputs[0].shape}")
        
        print("\n✅ Model Verification Complete!")
        print("\nNext steps:")
        print("1. Use whisper_char_map.json for token decoding")
        print("2. Input shape should be: batch_size × 80 mel bands × 3000 frames")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_existing_model()

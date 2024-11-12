import torch
import whisper
import onnx
import onnxruntime as ort
import numpy as np
from src.tokenizer_utils import WhisperTokenizer
import os

def test_existing_model():
    print("🔍 Testing Existing Whisper ONNX Model")
    
    # Get current directory path and construct ONNX model path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(current_dir, "tiny.onnx")
    
    try:
        # 1. Load and verify ONNX model
        print("\n1. Checking ONNX model...")
        print(f"Looking for model at: {onnx_path}")
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
        
        # 3. Create tokenizer
        print("\n3. Setting up tokenizer...")
        tokenizer = WhisperTokenizer("tiny")
        print("✓ Tokenizer initialized")
        
        # 4. Save character map
        print("\n4. Generating character map...")
        char_map_path = os.path.join(current_dir, "whisper_char_map.json")
        tokenizer.save_char_map(char_map_path)
        print(f"✓ Character map saved to {char_map_path}")
        
        # 5. Test with dummy input
        print("\n5. Testing inference...")
        dummy_input = np.random.randn(1, 80, 3000).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print("✓ Model successfully ran inference")
        print(f"Output shape: {outputs[0].shape}")
        
        print("\n✅ Model Verification Complete!")
        print(f"Your model at {onnx_path} is valid and ready for use.")
        print("\nNext steps:")
        print("1. Use whisper_char_map.json for token decoding")
        print("2. Input shape should be: batch_size × 80 mel bands × 3000 frames")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_existing_model()

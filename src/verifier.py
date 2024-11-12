import onnx
import numpy as np
import onnxruntime as ort
import torch
import whisper
from typing import Dict
import os

class WhisperONNXVerifier:
    @staticmethod
    def verify_model(onnx_path: str, pt_path: str) -> Dict:
        results = {}
        try:
            # Load and check ONNX model
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            results['model_valid'] = True
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            # Load PyTorch model
            checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))
            dims = checkpoint['dims']
            
            # Create test input
            test_input = np.random.randn(1, dims['n_mels'], dims['n_audio_ctx'] * 2).astype(np.float32)
            
            # Run ONNX inference
            ort_outputs = session.run(None, {input_name: test_input})
            
            # Run PyTorch inference
            torch_model = whisper.load_model("tiny")
            torch_model.load_state_dict(checkpoint['model_state_dict'])
            torch_model.eval()
            
            with torch.no_grad():
                torch_input = torch.from_numpy(test_input)
                torch_output = torch_model.encoder(torch_input)
            
            max_diff = np.max(np.abs(ort_outputs[0] - torch_output.numpy()))
            results['max_difference'] = float(max_diff)
            results['outputs_match'] = max_diff < 1e-4
            results['verification_success'] = True
            
        except Exception as e:
            results['verification_success'] = False
            results['error'] = str(e)
        
        return results

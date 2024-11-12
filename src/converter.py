import torch
import whisper
import warnings
import os
from typing import Optional, Tuple, Dict

class WhisperONNXConverter:
    def __init__(self, model_type: str = "tiny"):
        self.model_type = model_type
        self.model = None
        self.dims = None
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.dims = checkpoint['dims']
            self.model = whisper.load_model(self.model_type)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = whisper.load_model(self.model_type)
            self.dims = {
                'n_mels': 80,
                'n_audio_ctx': 1500,
                'n_audio_state': 384,
                'n_audio_head': 6,
                'n_audio_layer': 4
            }
        self.model.eval()
    
    def convert_to_onnx(self, output_path: str, input_shape: Optional[Tuple] = None) -> None:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if input_shape is None:
            input_shape = (1, self.dims['n_mels'], self.dims['n_audio_ctx'] * 2)
        
        dummy_input = torch.randn(*input_shape)
        
        with torch.no_grad():
            torch.onnx.export(
                self.model.encoder,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['mel_spectrogram'],
                output_names=['audio_features'],
                dynamic_axes={
                    'mel_spectrogram': {0: 'batch_size', 2: 'n_frames'},
                    'audio_features': {0: 'batch_size'}
                }
            )
        print(f"Model successfully converted and saved to: {output_path}")

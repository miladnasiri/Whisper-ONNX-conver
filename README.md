# Whisper ONNX Converter

Convert OpenAI's Whisper speech recognition models to ONNX format for deployment on Magic Leap 2 and other platforms.

## Features
- Convert Whisper models to ONNX format
- Support for all Whisper model sizes
- Verification tools to ensure conversion accuracy
- Optimized for real-time inference
- Unity/Magic Leap 2 deployment support

## Installation
```bash
git clone git@github.com:miladnasiri/Whisper-ONNX-conver.git
cd Whisper-ONNX-conver
pip install -r requirements.txt
```

## Quick Start
```python
from src.converter import WhisperONNXConverter
from src.verifier import WhisperONNXVerifier

# Initialize converter
converter = WhisperONNXConverter(model_type="tiny")
converter.load_model("path/to/your/model.pt")
converter.convert_to_onnx("output_model.onnx")
```

## Contact
Milad Nasiri

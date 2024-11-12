# Whisper ONNX Converter

# Whisper ONNX Converter

Convert OpenAI's Whisper speech recognition models to ONNX format for deployment on Magic Leap 2 and other platforms.

![Final Verification Check](https://github.com/miladnasiri/Whisper-ONNX-converter/blob/df9ea2ad57bcd3e88fdd07fbf1145d55b8d2041e/Final%20Verification%20Check.png)

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
[Milad Nasiri](https://www.linkedin.com/in/miladnasiri/)

## Verification
To verify your converted model:
```python
python src/final_verify.py
```
This will perform a comprehensive check of the ONNX model:
- Structure validation
- Inference testing
- PyTorch output comparison
- Size and performance analysis

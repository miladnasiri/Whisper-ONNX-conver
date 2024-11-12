# Whisper ONNX Converter

Convert OpenAI's Whisper speech recognition models to ONNX format for deployment on Magic Leap 2 and other platforms.
![Description of the image](https://github.com/miladnasiri/Whisper-ONNX-converter/blob/df81511c8897211d4760eac66da5223043378a00/Final%20Verification%20Check.png)

## Features
- Convert Whisper models to ONNX format
- Support for all Whisper model sizes
- Character map generation for token decoding
- Optimized for real-time inference
- Unity/Magic Leap 2 deployment support

## Installation
```bash
git clone git@github.com:miladnasiri/Whisper-ONNX-conver.git
cd Whisper-ONNX-conver
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Model Conversion
```python
from src.converter import WhisperONNXConverter
converter = WhisperONNXConverter("tiny")
converter.load_model()
converter.convert_to_onnx("whisper_tiny.onnx")
```

### Character Map Generation
```python
from src.tokenizer_utils import WhisperTokenizer
tokenizer = WhisperTokenizer("tiny")
tokenizer.save_char_map("whisper_char_map.json")
```

### Verification
```python
python test_existing_model.py
python check_char_map.py
```

## Model Details
- Input shape: `[batch_size, 80, n_frames]`
- Output shape: `[1, 1500, 384]`
- Character map size: 50,363 tokens

## Documentation
- [Character Map Documentation](docs/CHARACTER_MAP.md)

## Contact
Milad Nasiri

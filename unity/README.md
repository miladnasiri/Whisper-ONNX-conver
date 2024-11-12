# Unity Integration for Whisper ONNX

This directory contains Unity scripts for integrating the Whisper ONNX model with Magic Leap 2.

## Setup

1. Required Unity Packages:
   - Barracuda (for ONNX model)
   - Magic Leap XR Plugin
   - Newtonsoft JSON

2. Files needed:
   - ONNX model (`tiny.onnx`)
   - Character map (`whisper_char_map.json`)
   - Unity scripts in this directory

3. Unity Setup:
   - Create an empty GameObject
   - Add WhisperDecoder component
   - Assign ONNX model and character map
   - Add AudioCaptureML2 component
   - Link WhisperDecoder reference

## Usage
See the scripts for detailed usage information.

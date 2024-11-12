from src.converter import WhisperONNXConverter
from src.verifier import WhisperONNXVerifier

def main():
    # Initialize converter
    converter = WhisperONNXConverter(model_type="tiny")
    
    # Load and convert model
    converter.load_model("path/to/your/model.pt")
    converter.convert_to_onnx("output_model.onnx")
    
    # Verify conversion
    results = WhisperONNXVerifier.verify_model("output_model.onnx", "path/to/your/model.pt")
    print("Verification results:", results)

if __name__ == "__main__":
    main()

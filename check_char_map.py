import json
import os

def check_char_map():
    print("🔍 Checking character map contents...")
    
    # Get the full path to the character map file
    current_dir = os.getcwd()
    char_map_path = os.path.join(current_dir, "whisper_char_map.json")
    print(f"\nLooking for character map at: {char_map_path}")
    
    try:
        # Load the character map
        with open(char_map_path, 'r', encoding='utf-8') as f:
            char_map = json.load(f)
        
        print(f"\nTotal tokens in map: {len(char_map)}")
        
        # Check special tokens
        print("\nSpecial tokens found:")
        special_tokens = [
            "<|startoftranscript|>",
            "<|endoftext|>",
            "<|transcribe|>",
            "<|translate|>",
            "<|notimestamps|>"
        ]
        
        for token_id, token_text in char_map.items():
            if token_text in special_tokens:
                print(f"✓ {token_text} (ID: {token_id})")
        
        # Show sample regular tokens
        print("\nSample regular tokens:")
        sample_count = 0
        for token_id, token_text in char_map.items():
            if token_text not in special_tokens and sample_count < 5:
                print(f"Token {token_id}: '{token_text}'")
                sample_count += 1
        
        print("\n✅ Character map verification complete!")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Character map file not found at {char_map_path}")
        print("Please run the model verification script first to generate the character map.")
    except json.JSONDecodeError:
        print("\n❌ Error: Invalid JSON format in character map file")
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")

if __name__ == "__main__":
    check_char_map()

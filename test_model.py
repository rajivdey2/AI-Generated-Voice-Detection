"""
Test the trained model with a sample audio file

Usage:
    python test_model.py path/to/audio.mp3
"""

import sys
import json
import base64
from pathlib import Path

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
from src.model import VoiceDetectionModel

def encode_audio_to_base64(file_path):
    """Encode audio file to base64"""
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def test_single_file(audio_path):
    """Test model on a single audio file"""
    
    print("=" * 60)
    print("Testing AI Voice Detection Model")
    print("=" * 60)
    print(f"\nAudio file: {audio_path}")
    
    # Initialize components
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    model = VoiceDetectionModel()
    
    # Load model
    try:
        model.load()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease train the model first:")
        print("  python train.py")
        return
    
    # Process audio
    print("\nProcessing audio...")
    try:
        # Load and preprocess
        audio = processor.load_audio_file(audio_path)
        audio_processed = processor.preprocess_audio(audio)
        print(f"✓ Audio processed (shape: {audio_processed.shape})")
        
        # Extract features
        features = extractor.extract_all_features(audio_processed)
        print(f"✓ Features extracted ({len(features)} features)")
        
        # Make prediction
        result = model.predict(features)
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nClassification: {result['classification'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        print(f"  Human:        {result['human_probability']:.2%}")
        print(f"  AI-generated: {result['ai_probability']:.2%}")
        print(f"\nExplanation:")
        print(f"  {result['explanation']}")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

def test_api_format(audio_path):
    """Test with API-like format (base64)"""
    
    print("\n" + "=" * 60)
    print("Testing API Format (Base64)")
    print("=" * 60)
    
    # Initialize components
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    model = VoiceDetectionModel()
    
    # Load model
    try:
        model.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Encode audio to base64
    print("\nEncoding audio to Base64...")
    audio_base64 = encode_audio_to_base64(audio_path)
    print(f"✓ Encoded (length: {len(audio_base64)} characters)")
    
    # Process like API would
    print("\nProcessing through API pipeline...")
    try:
        # Decode and preprocess
        audio = processor.process_base64_audio(audio_base64)
        print(f"✓ Decoded and preprocessed")
        
        # Extract features
        features = extractor.extract_all_features(audio)
        print(f"✓ Features extracted")
        
        # Predict
        result = model.predict(features)
        
        # Display as JSON (like API response)
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_audio_file>")
        print("\nExample:")
        print("  python test_model.py data/raw/human/sample1.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Check if file exists
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Test with file directly
    test_single_file(audio_path)
    
    # Test with base64 (API format)
    test_api_format(audio_path)
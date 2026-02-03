"""
Training script for AI voice detection model

Usage:
    python train.py

Make sure you have audio samples in data/raw/ organized as:
    data/raw/human/
    data/raw/ai/
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
from src.model import VoiceDetectionModel
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_and_process_audio_files(data_dir, label, processor, extractor):
    """
    Load audio files, extract features
    
    Args:
        data_dir: Directory containing audio files
        label: 0 for human, 1 for AI
        processor: AudioProcessor instance
        extractor: FeatureExtractor instance
        
    Returns:
        features, labels
    """
    features_list = []
    labels_list = []
    
    # Get all audio files
    audio_files = list(Path(data_dir).glob("*.mp3")) + \
                  list(Path(data_dir).glob("*.wav")) + \
                  list(Path(data_dir).glob("*.m4a"))
    
    print(f"Processing {len(audio_files)} files from {data_dir}...")
    
    for audio_file in tqdm(audio_files):
        try:
            # Load and preprocess audio
            audio = processor.load_audio_file(str(audio_file))
            audio_processed = processor.preprocess_audio(audio)
            
            # Extract features
            features = extractor.extract_all_features(audio_processed)
            
            features_list.append(features)
            labels_list.append(label)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    return np.array(features_list), np.array(labels_list)

def main():
    """Main training function"""
    
    print("=" * 60)
    print("AI Voice Detection Model Training")
    print("=" * 60)
    
    # Initialize components
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    model = VoiceDetectionModel()
    
    # Define data directories
    human_dir = RAW_DATA_DIR / "human"
    ai_dir = RAW_DATA_DIR / "ai"
    
    # Check if directories exist
    if not human_dir.exists() or not ai_dir.exists():
        print("\nERROR: Data directories not found!")
        print(f"Please create these directories and add audio samples:")
        print(f"  - {human_dir}")
        print(f"  - {ai_dir}")
        print("\nOrganize your data as:")
        print("  data/raw/human/  (human voice samples)")
        print("  data/raw/ai/     (AI-generated voice samples)")
        return
    
    # Load and process data
    print("\n1. Loading Human Voice Samples...")
    X_human, y_human = load_and_process_audio_files(
        human_dir, 0, processor, extractor
    )
    
    print("\n2. Loading AI-Generated Voice Samples...")
    X_ai, y_ai = load_and_process_audio_files(
        ai_dir, 1, processor, extractor
    )
    
    # Combine datasets
    X = np.vstack([X_human, X_ai])
    y = np.concatenate([y_human, y_ai])
    
    print(f"\nTotal samples: {len(X)}")
    print(f"  - Human: {len(y_human)}")
    print(f"  - AI: {len(y_ai)}")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Check if we have enough data
    if len(X) < 20:
        print("\nWARNING: Very few samples! Add more data for better performance.")
        print("Recommended: At least 50-100 samples per class")
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    
    # Train model
    print("\n3. Training Model...")
    print("-" * 60)
    model.train(X_train, y_train, feature_names=feature_names)
    
    # Evaluate model
    print("\n4. Evaluating Model...")
    print("-" * 60)
    model.evaluate(X_test, y_test)
    
    # Save model
    print("\n5. Saving Model...")
    print("-" * 60)
    model.save()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nYou can now run the API with:")
    print("  uvicorn src.api:app --reload")
    print("\nOr test predictions with:")
    print("  python test_model.py")

if __name__ == "__main__":
    main()
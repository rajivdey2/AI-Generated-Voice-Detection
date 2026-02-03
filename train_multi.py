"""
Training script for multi-language AI voice detection
Trains separate models for each language

Usage:
    python train_multilingual.py
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
from src.model import VoiceDetectionModel
from config import RAW_DATA_DIR, MODEL_DIR

# Supported languages
LANGUAGES = {
    'english': 'en',
    'tamil': 'ta',
    'hindi': 'hi',
    'malayalam': 'ml',
    'telugu': 'te'
}

def load_language_specific_data(data_dir, label, language, processor, extractor):
    """
    Load audio files for a specific language
    
    Args:
        data_dir: Directory containing audio files
        label: 0 for human, 1 for AI
        language: Language name
        processor: AudioProcessor instance
        extractor: FeatureExtractor instance
        
    Returns:
        features, labels
    """
    features_list = []
    labels_list = []
    
    # Get audio files for this language
    audio_files = list(Path(data_dir).glob(f"{language}*.wav")) + \
                  list(Path(data_dir).glob(f"{language}*.mp3")) + \
                  list(Path(data_dir).glob(f"{language}_*.wav")) + \
                  list(Path(data_dir).glob(f"{language}_*.mp3"))
    
    print(f"  Found {len(audio_files)} {language} files")
    
    for audio_file in tqdm(audio_files, desc=f"  Processing {language}"):
        try:
            # Load and preprocess audio
            audio = processor.load_audio_file(str(audio_file))
            audio_processed = processor.preprocess_audio(audio)
            
            # Extract features
            features = extractor.extract_all_features(audio_processed)
            
            features_list.append(features)
            labels_list.append(label)
            
        except Exception as e:
            print(f"    Error processing {audio_file}: {e}")
            continue
    
    return np.array(features_list), np.array(labels_list)

def train_language_model(language, lang_code):
    """Train a model for a specific language"""
    
    print("\n" + "=" * 60)
    print(f"Training Model for {language.upper()} ({lang_code})")
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
        print(f"\nERROR: Data directories not found!")
        return False
    
    # Load language-specific data
    print(f"\n1. Loading {language} Human Voice Samples...")
    X_human, y_human = load_language_specific_data(
        human_dir, 0, language, processor, extractor
    )
    
    print(f"\n2. Loading {language} AI-Generated Voice Samples...")
    X_ai, y_ai = load_language_specific_data(
        ai_dir, 1, language, processor, extractor
    )
    
    # Check if we have data
    if len(X_human) == 0 or len(X_ai) == 0:
        print(f"\n⚠️ No data found for {language}. Skipping...")
        return False
    
    # Combine datasets
    X = np.vstack([X_human, X_ai])
    y = np.concatenate([y_human, y_ai])
    
    print(f"\nTotal {language} samples: {len(X)}")
    print(f"  - Human: {len(y_human)}")
    print(f"  - AI: {len(y_ai)}")
    
    # Check if we have enough data
    if len(X) < 10:
        print(f"\n⚠️ Too few samples for {language}! Need at least 10.")
        return False
    
    # Split into train/test sets
    test_size = 0.2 if len(X) >= 20 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    
    # Train model
    print(f"\n3. Training {language} Model...")
    print("-" * 60)
    model.train(X_train, y_train, feature_names=feature_names)
    
    # Evaluate model
    print(f"\n4. Evaluating {language} Model...")
    print("-" * 60)
    model.evaluate(X_test, y_test)
    
    # Save model
    model_path = MODEL_DIR / f"voice_detector_{lang_code}.pkl"
    scaler_path = MODEL_DIR / f"scaler_{lang_code}.pkl"
    
    print(f"\n5. Saving {language} Model...")
    print("-" * 60)
    model.save(model_path, scaler_path)
    
    return True

def train_universal_model():
    """Train a single model for all languages"""
    
    print("\n" + "=" * 60)
    print("Training Universal Multi-Language Model")
    print("=" * 60)
    
    # Initialize components
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    model = VoiceDetectionModel()
    
    # Define data directories
    human_dir = RAW_DATA_DIR / "human"
    ai_dir = RAW_DATA_DIR / "ai"
    
    # Load ALL audio files regardless of language
    print("\n1. Loading All Human Voice Samples...")
    audio_files = list(human_dir.glob("*.wav")) + list(human_dir.glob("*.mp3"))
    
    X_human_list = []
    for audio_file in tqdm(audio_files):
        try:
            audio = processor.load_audio_file(str(audio_file))
            audio_processed = processor.preprocess_audio(audio)
            features = extractor.extract_all_features(audio_processed)
            X_human_list.append(features)
        except Exception as e:
            print(f"  Error: {audio_file} - {e}")
    
    X_human = np.array(X_human_list)
    y_human = np.zeros(len(X_human))
    
    print(f"\n2. Loading All AI-Generated Voice Samples...")
    audio_files = list(ai_dir.glob("*.wav")) + list(ai_dir.glob("*.mp3"))
    
    X_ai_list = []
    for audio_file in tqdm(audio_files):
        try:
            audio = processor.load_audio_file(str(audio_file))
            audio_processed = processor.preprocess_audio(audio)
            features = extractor.extract_all_features(audio_processed)
            X_ai_list.append(features)
        except Exception as e:
            print(f"  Error: {audio_file} - {e}")
    
    X_ai = np.array(X_ai_list)
    y_ai = np.ones(len(X_ai))
    
    # Combine datasets
    X = np.vstack([X_human, X_ai])
    y = np.concatenate([y_human, y_ai])
    
    print(f"\nTotal samples (all languages): {len(X)}")
    print(f"  - Human: {len(y_human)}")
    print(f"  - AI: {len(y_ai)}")
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    feature_names = extractor.get_feature_names()
    
    print("\n3. Training Universal Model...")
    model.train(X_train, y_train, feature_names=feature_names)
    
    print("\n4. Evaluating Universal Model...")
    model.evaluate(X_test, y_test)
    
    print("\n5. Saving Universal Model...")
    model.save()
    
    return True

def main():
    """Main training function"""
    
    print("=" * 60)
    print("Multi-Language AI Voice Detection Training")
    print("=" * 60)
    
    print("\nChoose training mode:")
    print("1. Train separate models for each language (better accuracy)")
    print("2. Train single universal model (easier deployment)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Train language-specific models
        print("\n📋 Training separate models for each language...")
        
        success_count = 0
        for language, lang_code in LANGUAGES.items():
            if train_language_model(language, lang_code):
                success_count += 1
        
        print("\n" + "=" * 60)
        print(f"Training Complete! Successfully trained {success_count}/{len(LANGUAGES)} models")
        print("=" * 60)
        
    elif choice == "2":
        # Train universal model
        print("\n🌍 Training universal multi-language model...")
        
        if train_universal_model():
            print("\n" + "=" * 60)
            print("Universal Model Training Complete!")
            print("=" * 60)
    else:
        print("Invalid choice!")
        return
    
    print("\nYou can now run the API with:")
    print("  uvicorn src.api:app --reload")

if __name__ == "__main__":
    main()
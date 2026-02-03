"""
Generate synthetic audio samples for testing
NOTE: For production, you need REAL human and AI-generated voice samples
This script is just for testing the pipeline
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from config import RAW_DATA_DIR, SAMPLE_RATE

def generate_synthetic_voice(duration=5, is_ai=False):
    """
    Generate synthetic audio that simulates voice characteristics
    NOTE: This is just for testing - use real data for production!
    
    Args:
        duration: Length in seconds
        is_ai: If True, add artifacts typical of AI voices
    """
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples)
    
    # Base frequency (fundamental frequency of voice)
    f0 = 150 if np.random.random() > 0.5 else 250  # Male/Female
    
    # Generate harmonics (voice has multiple harmonics)
    signal = np.zeros(samples)
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic
        frequency = f0 * harmonic
        
        if is_ai:
            # AI voices often have more regular patterns
            phase = 0
            freq_variation = 0.1  # Less variation
        else:
            # Human voices have natural variation
            phase = np.random.random() * 2 * np.pi
            freq_variation = 0.3  # More variation
        
        # Add frequency modulation (vibrato)
        vibrato = freq_variation * np.sin(2 * np.pi * 5 * t)
        signal += amplitude * np.sin(2 * np.pi * frequency * t * (1 + vibrato) + phase)
    
    # Add formants (resonances typical of human voice)
    for formant_freq in [800, 1200, 2500]:
        formant = 0.3 * np.sin(2 * np.pi * formant_freq * t)
        signal += formant
    
    # Add noise (breath, etc.)
    if is_ai:
        noise_level = 0.01  # AI voices typically have less noise
    else:
        noise_level = 0.05  # Human voices have more natural noise
    
    signal += noise_level * np.random.randn(samples)
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Add pauses (speech-like patterns)
    pause_mask = np.ones(samples)
    n_pauses = np.random.randint(2, 5)
    for _ in range(n_pauses):
        pause_start = np.random.randint(0, samples - SAMPLE_RATE)
        pause_duration = np.random.randint(int(0.1 * SAMPLE_RATE), int(0.3 * SAMPLE_RATE))
        pause_mask[pause_start:pause_start + pause_duration] = 0.1
    
    signal = signal * pause_mask
    
    return signal

def generate_dataset(n_samples_per_class=10):
    """Generate synthetic dataset for testing"""
    
    print("=" * 60)
    print("Generating Synthetic Test Data")
    print("=" * 60)
    print("\nWARNING: This is synthetic data for testing only!")
    print("For production, use REAL human and AI-generated voice samples.")
    print("=" * 60)
    
    # Create directories
    human_dir = RAW_DATA_DIR / "human"
    ai_dir = RAW_DATA_DIR / "ai"
    
    human_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate human samples
    print(f"\nGenerating {n_samples_per_class} human voice samples...")
    for i in range(n_samples_per_class):
        audio = generate_synthetic_voice(duration=5, is_ai=False)
        filename = human_dir / f"human_sample_{i+1:03d}.wav"
        sf.write(filename, audio, SAMPLE_RATE)
        print(f"  Created: {filename.name}")
    
    # Generate AI samples
    print(f"\nGenerating {n_samples_per_class} AI voice samples...")
    for i in range(n_samples_per_class):
        audio = generate_synthetic_voice(duration=5, is_ai=True)
        filename = ai_dir / f"ai_sample_{i+1:03d}.wav"
        sf.write(filename, audio, SAMPLE_RATE)
        print(f"  Created: {filename.name}")
    
    print("\n" + "=" * 60)
    print("Sample Data Generation Complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  Human samples: {human_dir}")
    print(f"  AI samples: {ai_dir}")
    print(f"\nNext steps:")
    print("  1. Replace with REAL audio samples")
    print("  2. Run: python train.py")
    print("  3. Run: uvicorn src.api:app --reload")

if __name__ == "__main__":
    generate_dataset(n_samples_per_class=20)
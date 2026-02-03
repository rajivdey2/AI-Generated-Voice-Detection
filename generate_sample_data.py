"""
Generate synthetic multi-language audio samples for testing
Generates samples for: Tamil, English, Hindi, Malayalam, Telugu
NOTE: For production, use REAL human and AI-generated voice samples
This script is just for testing the pipeline
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from config import RAW_DATA_DIR, SAMPLE_RATE

# Language-specific characteristics
LANGUAGE_CONFIGS = {
    'english': {
        'base_freq_male': 120,
        'base_freq_female': 220,
        'formants': [800, 1200, 2500, 3500],
        'syllable_rate': 4.5,  # syllables per second
        'pitch_variation': 0.25
    },
    'tamil': {
        'base_freq_male': 110,
        'base_freq_female': 210,
        'formants': [700, 1100, 2400, 3300],
        'syllable_rate': 5.2,  # Tamil is slightly faster
        'pitch_variation': 0.30
    },
    'hindi': {
        'base_freq_male': 115,
        'base_freq_female': 215,
        'formants': [750, 1150, 2450, 3400],
        'syllable_rate': 4.8,
        'pitch_variation': 0.28
    },
    'malayalam': {
        'base_freq_male': 118,
        'base_freq_female': 218,
        'formants': [780, 1180, 2480, 3450],
        'syllable_rate': 5.0,
        'pitch_variation': 0.27
    },
    'telugu': {
        'base_freq_male': 112,
        'base_freq_female': 212,
        'formants': [720, 1120, 2420, 3350],
        'syllable_rate': 4.9,
        'pitch_variation': 0.29
    }
}

def generate_synthetic_voice(duration=5, is_ai=False, language='english', gender='random'):
    """
    Generate synthetic audio that simulates voice characteristics
    
    Args:
        duration: Length in seconds
        is_ai: If True, add artifacts typical of AI voices
        language: Language name (english, tamil, hindi, malayalam, telugu)
        gender: 'male', 'female', or 'random'
    
    Returns:
        numpy array of audio samples
    """
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples)
    
    # Get language config
    lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['english'])
    
    # Determine gender
    if gender == 'random':
        gender = 'male' if np.random.random() > 0.5 else 'female'
    
    # Base frequency (fundamental frequency)
    if gender == 'male':
        f0 = lang_config['base_freq_male']
    else:
        f0 = lang_config['base_freq_female']
    
    # Add natural variation to base frequency
    f0 += np.random.uniform(-5, 5)
    
    # Generate harmonics (voice has multiple harmonics)
    signal = np.zeros(samples)
    
    for harmonic in range(1, 8):
        amplitude = 1.0 / harmonic
        frequency = f0 * harmonic
        
        if is_ai:
            # AI voices: more regular, less variation
            phase = 0
            freq_variation = 0.05  # Very little variation
            amplitude_variation = 0.02
        else:
            # Human voices: natural variation
            phase = np.random.random() * 2 * np.pi
            freq_variation = lang_config['pitch_variation'] * np.random.uniform(0.8, 1.2)
            amplitude_variation = 0.1
        
        # Add frequency modulation (vibrato) - natural pitch variations
        vibrato_rate = np.random.uniform(4, 7)  # Hz
        vibrato = freq_variation * np.sin(2 * np.pi * vibrato_rate * t)
        
        # Add amplitude modulation - speech rhythm
        rhythm_rate = lang_config['syllable_rate']
        rhythm = 1 + amplitude_variation * np.sin(2 * np.pi * rhythm_rate * t)
        
        # Generate harmonic
        harmonic_signal = amplitude * rhythm * np.sin(
            2 * np.pi * frequency * t * (1 + vibrato) + phase
        )
        signal += harmonic_signal
    
    # Add formants (resonances typical of human voice for this language)
    for formant_freq in lang_config['formants']:
        # Formant bandwidth
        bandwidth = np.random.uniform(50, 150)
        formant_amplitude = 0.2 / (1 + (formant_freq / 1000))
        
        # Add formant resonance
        formant = formant_amplitude * np.sin(2 * np.pi * formant_freq * t)
        formant *= np.exp(-bandwidth * t / SAMPLE_RATE)  # Decay
        signal += formant
    
    # Add noise (breath, aspiration, etc.)
    if is_ai:
        noise_level = 0.005  # AI voices: very clean
        noise_color = 'white'  # More uniform
    else:
        noise_level = 0.03  # Human voices: more noise
        noise_color = 'pink'  # More natural spectrum
    
    # Generate noise
    if noise_color == 'pink':
        # Pink noise (1/f spectrum - more natural)
        white_noise = np.random.randn(samples)
        # Simple pink noise approximation
        noise = np.cumsum(white_noise) / np.sqrt(samples)
        noise = noise / np.std(noise)
    else:
        # White noise
        noise = np.random.randn(samples)
    
    signal += noise_level * noise
    
    # Create speech-like patterns with pauses
    syllable_duration = 1.0 / lang_config['syllable_rate']
    n_syllables = int(duration / syllable_duration)
    
    envelope = np.ones(samples)
    
    for i in range(n_syllables):
        # Random syllable start time
        syllable_start = int(i * syllable_duration * SAMPLE_RATE)
        syllable_end = int((i + 0.7) * syllable_duration * SAMPLE_RATE)
        
        if syllable_end > samples:
            break
        
        # Create syllable envelope (attack-sustain-release)
        syllable_len = syllable_end - syllable_start
        attack = int(syllable_len * 0.1)
        release = int(syllable_len * 0.2)
        
        # Attack
        envelope[syllable_start:syllable_start + attack] = np.linspace(0.3, 1.0, attack)
        # Release
        if syllable_end - release < samples:
            envelope[syllable_end - release:syllable_end] = np.linspace(1.0, 0.2, release)
        
        # Random pause after some syllables
        if np.random.random() > 0.7 and i < n_syllables - 1:
            pause_duration = int(np.random.uniform(0.1, 0.3) * SAMPLE_RATE)
            pause_start = syllable_end
            pause_end = min(pause_start + pause_duration, samples)
            envelope[pause_start:pause_end] = 0.1
    
    # Apply envelope
    signal = signal * envelope
    
    # Add language-specific intonation patterns
    if not is_ai:
        # Natural intonation contour
        intonation_freq = 0.5  # Overall pitch contour frequency
        intonation = 0.15 * np.sin(2 * np.pi * intonation_freq * t + np.random.random())
        signal = signal * (1 + intonation)
    
    # Normalize
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Add final processing differences
    if is_ai:
        # AI voices: slight compression (more uniform amplitude)
        threshold = 0.5
        signal = np.where(np.abs(signal) > threshold, 
                         threshold + (np.abs(signal) - threshold) * 0.3,
                         signal)
    else:
        # Human voices: natural dynamics
        pass
    
    return signal.astype(np.float32)

def generate_multilingual_dataset(n_samples_per_language_per_class=20):
    """Generate synthetic multilingual dataset for testing"""
    
    print("=" * 70)
    print("Generating Multi-Language Synthetic Test Data")
    print("=" * 70)
    print("\n⚠️  WARNING: This is synthetic data for TESTING ONLY!")
    print("For production/hackathon, use REAL human and AI-generated samples.")
    print("=" * 70)
    
    # Create directories
    human_dir = RAW_DATA_DIR / "human"
    ai_dir = RAW_DATA_DIR / "ai"
    
    human_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)
    
    languages = ['english', 'tamil', 'hindi', 'malayalam', 'telugu']
    total_samples = len(languages) * n_samples_per_language_per_class * 2
    
    print(f"\n📊 Generating {n_samples_per_language_per_class} samples per language per class")
    print(f"   Total: {total_samples} samples ({len(languages)} languages × 2 classes)\n")
    
    sample_count = 0
    
    for language in languages:
        print(f"\n🌍 Generating {language.upper()} samples...")
        print("-" * 70)
        
        # Generate human samples
        print(f"  👤 Human voices ({n_samples_per_language_per_class} samples)...")
        for i in range(n_samples_per_language_per_class):
            # Vary duration slightly
            duration = np.random.uniform(4, 6)
            
            # Mix male and female voices
            gender = 'male' if i % 2 == 0 else 'female'
            
            audio = generate_synthetic_voice(
                duration=duration, 
                is_ai=False, 
                language=language,
                gender=gender
            )
            
            filename = human_dir / f"{language}_human_{i+1:03d}.wav"
            sf.write(filename, audio, SAMPLE_RATE)
            sample_count += 1
            
            if (i + 1) % 5 == 0:
                print(f"     ✓ Generated {i+1}/{n_samples_per_language_per_class} samples")
        
        # Generate AI samples
        print(f"  🤖 AI-generated voices ({n_samples_per_language_per_class} samples)...")
        for i in range(n_samples_per_language_per_class):
            # Vary duration slightly
            duration = np.random.uniform(4, 6)
            
            # Mix male and female voices
            gender = 'male' if i % 2 == 0 else 'female'
            
            audio = generate_synthetic_voice(
                duration=duration, 
                is_ai=True, 
                language=language,
                gender=gender
            )
            
            filename = ai_dir / f"{language}_ai_{i+1:03d}.wav"
            sf.write(filename, audio, SAMPLE_RATE)
            sample_count += 1
            
            if (i + 1) % 5 == 0:
                print(f"     ✓ Generated {i+1}/{n_samples_per_language_per_class} samples")
        
        print(f"  ✅ {language.upper()} complete: {n_samples_per_language_per_class * 2} samples")
    
    print("\n" + "=" * 70)
    print("✅ Sample Data Generation Complete!")
    print("=" * 70)
    
    print(f"\n📁 Generated {sample_count} total samples:")
    print(f"   Location: {RAW_DATA_DIR}")
    print(f"   • Human samples: {human_dir}")
    print(f"   • AI samples: {ai_dir}")
    
    print("\n📊 Distribution:")
    for language in languages:
        human_count = len(list(human_dir.glob(f"{language}_*.wav")))
        ai_count = len(list(ai_dir.glob(f"{language}_*.wav")))
        print(f"   • {language.capitalize():12s}: {human_count} human + {ai_count} AI = {human_count + ai_count} total")
    
    print("\n⚠️  IMPORTANT REMINDER:")
    print("   This synthetic data is for pipeline testing only!")
    print("   For the hackathon, replace with REAL audio samples:")
    print("   • Human: Record yourself or use datasets like Common Voice")
    print("   • AI: Generate using ElevenLabs, Google TTS, Azure TTS, etc.")
    
    print("\n🚀 Next steps:")
    print("   1. [RECOMMENDED] Replace with real audio samples")
    print("   2. Train model: python train_multilingual.py")
    print("   3. Test API: uvicorn src.api:app --reload")
    print("   4. Deploy for hackathon")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Generate samples
    # Adjust this number based on your needs:
    # - 10-20: Quick test
    # - 50: Basic hackathon demo
    # - 100+: Better model performance
    
    n_samples = 50  # 50 samples per language per class = 500 total samples
    
    print(f"\n🎯 Generating {n_samples} samples per language per class...")
    print("   (You can change this in the script)\n")
    
    generate_multilingual_dataset(n_samples_per_language_per_class=n_samples)
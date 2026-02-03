import base64
import io
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, DURATION

class AudioProcessor:
    """Handles audio loading, decoding, and preprocessing"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_samples = sample_rate * duration
    
    def decode_base64_audio(self, base64_string):
        """
        Decode Base64-encoded audio (WAV/MP3) to audio array
        
        Args:
            base64_string: Base64 encoded audio string
            
        Returns:
            numpy array of audio samples
        """
        try:
            # Remove header if present (data:audio/mp3;base64,)
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            audio_bytes = base64.b64decode(base64_string)
            
            # Try to detect format and load accordingly
            audio_array = None
            sr = None
            
            # First try: Direct load with soundfile (works for WAV)
            try:
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            except:
                # Second try: Use pydub for MP3/other formats
                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                    wav_io = io.BytesIO()
                    audio.export(wav_io, format='wav')
                    wav_io.seek(0)
                    audio_array, sr = sf.read(wav_io)
                except Exception as e:
                    raise ValueError(f"Could not decode audio format: {str(e)}")
            
            if audio_array is None:
                raise ValueError("Failed to decode audio")
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sr, 
                    target_sr=self.sample_rate
                )
            
            return audio_array
            
        except Exception as e:
            raise ValueError(f"Error decoding audio: {str(e)}")
    
    def load_audio_file(self, file_path):
        """
        Load audio from file path
        
        Args:
            file_path: Path to audio file
            
        Returns:
            numpy array of audio samples
        """
        try:
            audio_array, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio_array
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
    
    def preprocess_audio(self, audio_array):
        """
        Preprocess audio: normalize, trim silence, pad/truncate
        
        Args:
            audio_array: Raw audio numpy array
            
        Returns:
            Preprocessed audio array
        """
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Trim silence from beginning and end
        audio_array, _ = librosa.effects.trim(
            audio_array, 
            top_db=20
        )
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Pad or truncate to fixed length
        if len(audio_array) > self.max_samples:
            # Truncate
            audio_array = audio_array[:self.max_samples]
        else:
            # Pad with zeros
            padding = self.max_samples - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant')
        
        return audio_array
    
    def process_base64_audio(self, base64_string):
        """
        Complete pipeline: decode -> preprocess
        
        Args:
            base64_string: Base64 encoded audio
            
        Returns:
            Preprocessed audio array
        """
        audio_array = self.decode_base64_audio(base64_string)
        processed_audio = self.preprocess_audio(audio_array)
        return processed_audio


# Example usage
if __name__ == "__main__":
    processor = AudioProcessor()
    
    # Test with a file
    # audio = processor.load_audio_file("path/to/audio.mp3")
    # processed = processor.preprocess_audio(audio)
    # print(f"Processed audio shape: {processed.shape}")
    
    print("AudioProcessor module loaded successfully")
    print(f"Sample rate: {processor.sample_rate}")
    print(f"Duration: {processor.duration}s")
    print(f"Max samples: {processor.max_samples}")
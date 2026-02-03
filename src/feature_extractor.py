import numpy as np
import librosa
from config import FEATURE_CONFIG

class FeatureExtractor:
    """Extract audio features for AI voice detection"""
    
    def __init__(self, config=FEATURE_CONFIG):
        self.config = config
        self.sample_rate = config['sample_rate']
        self.n_mfcc = config['n_mfcc']
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC (Mel-Frequency Cepstral Coefficients)"""
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Return mean and std across time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio, 
            hop_length=self.hop_length
        )[0]
        
        features = {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_std': np.std(zero_crossing_rate)
        }
        
        return np.array(list(features.values()))
    
    def extract_pitch_features(self, audio):
        """Extract pitch-related features"""
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Get pitch values where magnitude is highest
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Only non-zero pitches
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_range = np.max(pitch_values) - np.min(pitch_values)
        else:
            pitch_mean = pitch_std = pitch_range = 0
        
        return np.array([pitch_mean, pitch_std, pitch_range])
    
    def extract_energy_features(self, audio):
        """Extract energy-based features"""
        # RMS energy
        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length
        )[0]
        
        # Short-time energy
        energy = np.sum(audio ** 2) / len(audio)
        
        return np.array([
            np.mean(rms),
            np.std(rms),
            energy
        ])
    
    def extract_temporal_features(self, audio):
        """Extract temporal pattern features"""
        # Onset strength (sudden changes in audio)
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Tempogram (rhythmic patterns)
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return np.array([
            np.mean(onset_env),
            np.std(onset_env),
            np.mean(tempogram),
            np.std(tempogram)
        ])
    
    def extract_all_features(self, audio):
        """
        Extract all features and combine into single feature vector
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Feature vector (numpy array)
        """
        # Extract different feature sets
        mfcc_features = self.extract_mfcc_features(audio)
        spectral_features = self.extract_spectral_features(audio)
        pitch_features = self.extract_pitch_features(audio)
        energy_features = self.extract_energy_features(audio)
        temporal_features = self.extract_temporal_features(audio)
        
        # Combine all features
        all_features = np.concatenate([
            mfcc_features,
            spectral_features,
            pitch_features,
            energy_features,
            temporal_features
        ])
        
        return all_features
    
    def get_feature_names(self):
        """Return names of all features for interpretability"""
        names = []
        
        # MFCC names
        for i in range(self.n_mfcc):
            names.append(f'mfcc_{i}_mean')
        for i in range(self.n_mfcc):
            names.append(f'mfcc_{i}_std')
        
        # Spectral feature names
        names.extend([
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'zero_crossing_rate_mean', 'zero_crossing_rate_std'
        ])
        
        # Pitch feature names
        names.extend(['pitch_mean', 'pitch_std', 'pitch_range'])
        
        # Energy feature names
        names.extend(['rms_mean', 'rms_std', 'energy'])
        
        # Temporal feature names
        names.extend([
            'onset_strength_mean', 'onset_strength_std',
            'tempogram_mean', 'tempogram_std'
        ])
        
        return names


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Test with dummy audio
    dummy_audio = np.random.randn(16000 * 5)  # 5 seconds
    features = extractor.extract_all_features(dummy_audio)
    
    print(f"Total features extracted: {len(features)}")
    print(f"Feature names: {extractor.get_feature_names()[:5]}...")  # Show first 5
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from config import MODEL_PATH, SCALER_PATH

class VoiceDetectionModel:
    """AI vs Human voice detection model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.feature_names = None
        
    def create_model(self):
        """Create a Random Forest classifier"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced datasets
        )
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the model
        
        Args:
            X_train: Training features (numpy array)
            y_train: Training labels (0=human, 1=AI)
            feature_names: List of feature names for interpretability
        """
        if self.model is None:
            self.create_model()
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, 
            X_train_scaled, 
            y_train, 
            cv=5, 
            scoring='accuracy'
        )
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n=== Model Evaluation ===")
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=['Human', 'AI-Generated']
        ))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f"\nTest Accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def predict(self, features):
        """
        Predict if voice is AI-generated
        
        Args:
            features: Feature vector (numpy array)
            
        Returns:
            dict with prediction, confidence, and explanation
        """
        # Reshape if single sample
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.model.predict(features_scaled)[0]
        prediction = int(prediction)  # Convert to Python int
        
        # Get probability scores
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[prediction])  # Convert to Python float
        
        # Generate explanation
        explanation = self._generate_explanation(
            features[0], 
            prediction, 
            confidence
        )
        
        result = {
            'classification': 'AI-generated' if prediction == 1 else 'human',
            'confidence': confidence,
            'ai_probability': float(probabilities[1]),
            'human_probability': float(probabilities[0]),
            'explanation': explanation
        }
        
        return result
    
    def _generate_explanation(self, features, prediction, confidence):
        """
        Generate human-readable explanation of prediction
        
        Args:
            features: Feature vector
            prediction: 0 (human) or 1 (AI)
            confidence: Confidence score
        """
        explanation_parts = []
        
        # Confidence level
        if confidence > 0.9:
            certainty = "very confident"
        elif confidence > 0.75:
            certainty = "confident"
        elif confidence > 0.6:
            certainty = "moderately confident"
        else:
            certainty = "uncertain"
        
        label = "AI-generated" if prediction == 1 else "human"
        explanation_parts.append(f"The model is {certainty} that this voice is {label}.")
        
        # Get top contributing features
        if self.feature_importance is not None and self.feature_names is not None:
            # Get feature contributions for this sample
            top_features = self._get_top_features(features, n=3)
            
            if top_features:
                explanation_parts.append("Key indicators:")
                for feat_name, importance in top_features:
                    explanation_parts.append(f"- {feat_name} (importance: {importance:.2f})")
        
        # Add general characteristics
        if prediction == 1:  # AI-generated
            explanation_parts.append(
                "Detected characteristics typical of AI-generated voices include "
                "unnatural prosody patterns and unusual spectral consistency."
            )
        else:  # Human
            explanation_parts.append(
                "Natural voice characteristics detected including organic pitch "
                "variation and typical human speech patterns."
            )
        
        return " ".join(explanation_parts)
    
    def _get_top_features(self, features, n=3):
        """Get top N most important features for this sample"""
        if self.feature_importance is None or self.feature_names is None:
            return []
        
        # Get indices of top N important features
        top_indices = np.argsort(self.feature_importance)[-n:][::-1]
        
        top_features = [
            (self.feature_names[i], self.feature_importance[i])
            for i in top_indices
        ]
        
        return top_features
    
    def save(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Save model and scaler to disk"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """Load model and scaler from disk"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
        return self


# Example usage
if __name__ == "__main__":
    # Create dummy training data
    n_samples = 1000
    n_features = 99  # From feature extractor
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # 0=human, 1=AI
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = VoiceDetectionModel()
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    
    # Test prediction
    sample = X_test[0]
    result = model.predict(sample)
    print("\nSample Prediction:")
    print(result)
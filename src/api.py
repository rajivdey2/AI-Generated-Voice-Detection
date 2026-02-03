from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
from src.model import VoiceDetectionModel
from config import MODEL_PATH, SCALER_PATH

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Multi-language AI-generated voice detection system",
    version="1.0.0"
)

# Initialize components
audio_processor = AudioProcessor()
feature_extractor = FeatureExtractor()
model = VoiceDetectionModel()

# Load trained model (will be trained separately)
try:
    model.load(MODEL_PATH, SCALER_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    print("Please train the model first using train.py")

# Request/Response models
class AudioInput(BaseModel):
    audio_base64: str = Field(
        ..., 
        description="Base64-encoded MP3 audio file"
    )
    language: Optional[str] = Field(
        None,
        description="Language of the audio (tamil, english, hindi, malayalam, telugu)"
    )

class DetectionResponse(BaseModel):
    classification: str = Field(
        ..., 
        description="Classification result: 'AI-generated' or 'human'"
    )
    confidence: float = Field(
        ..., 
        description="Confidence score between 0 and 1"
    )
    ai_probability: float = Field(
        ...,
        description="Probability of being AI-generated"
    )
    human_probability: float = Field(
        ...,
        description="Probability of being human"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the decision"
    )
    language: Optional[str] = Field(
        None,
        description="Language of the audio (if provided)"
    )

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Voice Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Detect if voice is AI-generated",
            "/health": "GET - Check API health status",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "components": {
            "audio_processor": True,
            "feature_extractor": True,
            "model": model_loaded
        }
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_ai_voice(audio_input: AudioInput):
    """
    Detect if a voice sample is AI-generated or human
    
    Args:
        audio_input: AudioInput object with base64-encoded MP3
        
    Returns:
        DetectionResponse with classification, confidence, and explanation
    """
    try:
        # Check if model is loaded
        if model.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first."
            )
        
        # Step 1: Decode and preprocess audio
        try:
            audio_array = audio_processor.process_base64_audio(
                audio_input.audio_base64
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio data: {str(e)}"
            )
        
        # Step 2: Extract features
        try:
            features = feature_extractor.extract_all_features(audio_array)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Feature extraction failed: {str(e)}"
            )
        
        # Step 3: Make prediction
        try:
            result = model.predict(features)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
        
        # Add language to response if provided
        if audio_input.language:
            result['language'] = audio_input.language
        
        return DetectionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
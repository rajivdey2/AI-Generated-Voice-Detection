from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
from src.model import VoiceDetectionModel
from config import MODEL_PATH, SCALER_PATH, MODEL_DIR

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Multi-language AI-generated voice detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get configuration from environment variables
API_KEY = os.getenv("API_KEY", "hackathon_demo_key_2024")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
PORT = int(os.getenv("PORT", 8000))

print(f"🚀 Starting API in {ENVIRONMENT} mode")
print(f"🔑 API Key configured: {'Yes' if API_KEY else 'No'}")

# Language mapping
LANGUAGE_CODES = {
    'english': 'en', 'en': 'en',
    'tamil': 'ta', 'ta': 'ta',
    'hindi': 'hi', 'hi': 'hi',
    'malayalam': 'ml', 'ml': 'ml',
    'telugu': 'te', 'te': 'te'
}

# Initialize components
audio_processor = AudioProcessor()
feature_extractor = FeatureExtractor()

# Load models - try language-specific first, fall back to universal
models = {}
universal_model = None

print("Loading models...")
# Try to load universal model
try:
    universal_model = VoiceDetectionModel()
    universal_model.load(MODEL_PATH, SCALER_PATH)
    print("✓ Universal model loaded")
except Exception as e:
    print(f"⚠️ Universal model not found: {e}")

# Try to load language-specific models
for lang_name, lang_code in LANGUAGE_CODES.items():
    if lang_code not in models:  # Avoid duplicates
        try:
            model_path = MODEL_DIR / f"voice_detector_{lang_code}.pkl"
            scaler_path = MODEL_DIR / f"scaler_{lang_code}.pkl"
            
            if model_path.exists():
                model = VoiceDetectionModel()
                model.load(model_path, scaler_path)
                models[lang_code] = model
                print(f"✓ {lang_name} ({lang_code}) model loaded")
        except Exception as e:
            print(f"⚠️ {lang_name} model not found: {e}")

if not models and not universal_model:
    print("⚠️ WARNING: No models loaded! Please train models first.")
else:
    print(f"✓ Ready with {len(models)} language-specific models" + 
          (" and 1 universal model" if universal_model else ""))

def get_model_for_language(language):
    """Get the appropriate model for a language"""
    if not language:
        # No language specified, use universal if available
        return universal_model if universal_model else list(models.values())[0] if models else None
    
    # Normalize language code
    lang_code = LANGUAGE_CODES.get(language.lower())
    
    if not lang_code:
        # Unknown language, use universal
        return universal_model if universal_model else list(models.values())[0] if models else None
    
    # Return language-specific model if available, otherwise universal
    return models.get(lang_code, universal_model)

# Request/Response models
class AudioInput(BaseModel):
    audio_base64: str = Field(
        ..., 
        description="Base64-encoded audio file (WAV/MP3)",
        alias="audio_base64"
    )
    language: Optional[str] = Field(
        None,
        description="Language of the audio (tamil, english, hindi, malayalam, telugu)",
        alias="language"
    )
    audio_format: Optional[str] = Field(
        "wav",
        description="Audio format (wav or mp3)",
        alias="audio_format"
    )
    
    class Config:
        # Allow field aliases and extra fields
        populate_by_name = True
        extra = "ignore"  # Ignore unknown fields
        # Make field order flexible
        json_schema_extra = {
            "example": {
                "audio_base64": "UklGRiQAAABXQVZFZm10...",
                "language": "en",
                "audio_format": "wav"
            }
        }

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

# Helper function to verify API key
def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key (optional in development mode)"""
    # In development mode, allow requests without API key
    if ENVIRONMENT == "development" and x_api_key is None:
        return "dev_bypass"
    
    # In production, require API key
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'x-api-key' header in your request."
        )
    
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return x_api_key

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Voice Detection API",
        "version": "1.0.0",
        "status": "running",
        "environment": ENVIRONMENT,
        "endpoints": {
            "/detect": "POST - Detect if voice is AI-generated (requires API key)",
            "/predict": "POST - Alias for /detect (requires API key)",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        },
        "supported_languages": ["en", "ta", "hi", "ml", "te"],
        "authentication": "Include 'x-api-key' header in requests"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy" if (models or universal_model) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "environment": ENVIRONMENT,
        "models_loaded": {
            "universal": universal_model is not None,
            "language_specific": list(models.keys()) if models else [],
            "total_models": len(models) + (1 if universal_model else 0)
        },
        "supported_languages": list(LANGUAGE_CODES.values()),
        "components": {
            "audio_processor": True,
            "feature_extractor": True,
            "api_version": "1.0.0"
        }
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_ai_voice(
    audio_input: AudioInput,
    x_api_key: str = Header(None, description="API Key for authentication (optional in dev mode)")
):
    """
    Detect if a voice sample is AI-generated or human
    
    Accepts flexible JSON input - field order doesn't matter:
    ```json
    {
        "audio_base64": "base64_string",
        "language": "en",
        "audio_format": "wav"
    }
    ```
    
    Alternative field names also supported:
    - audio_base64 / audio / base64
    - language / lang / language_code
    - audio_format / format / file_format
    
    Args:
        audio_input: AudioInput object with base64-encoded audio
        x_api_key: API key in header
        
    Returns:
        DetectionResponse with classification, confidence, and explanation
    """
    # Verify API key
    verify_api_key(x_api_key)
    
    try:
        # Get appropriate model for language
        model = get_model_for_language(audio_input.language)
        
        # Check if model is available
        if model is None or model.model is None:
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

# Alias endpoint for hackathon (some use /predict instead of /detect)
@app.post("/predict", response_model=DetectionResponse)
async def predict_ai_voice(
    audio_input: AudioInput,
    x_api_key: str = Header(None, description="API Key for authentication (optional in dev mode)")
):
    """Alias for /detect endpoint"""
    return await detect_ai_voice(audio_input, x_api_key)

# Flexible endpoint that handles various field names
@app.post("/api/detect", response_model=DetectionResponse)
@app.post("/api/predict", response_model=DetectionResponse)
async def flexible_detect(
    request: dict,
    x_api_key: str = Header(None, description="API Key for authentication")
):
    """
    Flexible detection endpoint that accepts various field name formats
    
    Accepts any of these field name variations:
    - audio_base64 / audio / base64 / audio_data
    - language / lang / language_code / ln
    - audio_format / format / file_format / type
    """
    verify_api_key(x_api_key)
    
    # Extract audio_base64 with flexible field names
    audio_base64 = (
        request.get('audio_base64') or 
        request.get('audio') or 
        request.get('base64') or 
        request.get('audio_data')
    )
    
    if not audio_base64:
        raise HTTPException(
            status_code=400,
            detail="Missing audio data. Provide one of: audio_base64, audio, base64, audio_data"
        )
    
    # Extract language with flexible field names
    language = (
        request.get('language') or 
        request.get('lang') or 
        request.get('language_code') or 
        request.get('ln')
    )
    
    # Extract format with flexible field names
    audio_format = (
        request.get('audio_format') or 
        request.get('format') or 
        request.get('file_format') or 
        request.get('type') or 
        'wav'
    )
    
    # Create normalized AudioInput
    normalized_input = AudioInput(
        audio_base64=audio_base64,
        language=language,
        audio_format=audio_format
    )
    
    return await detect_ai_voice(normalized_input, x_api_key)

# Run with: uvicorn src.api:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
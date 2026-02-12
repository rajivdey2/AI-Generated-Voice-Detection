
# 🎙️ AI Voice Detection API

Multi-language AI-generated voice detection system for Tamil, English, Hindi, Malayalam, and Telugu.

## 🚀 Quick Start

### API Endpoints

- **POST** `/detect` - Detect if voice is AI-generated
- **POST** `/predict` - Alternative endpoint (same as /detect)
- **POST** `/api/detect` - Flexible endpoint (accepts various field names)
- **GET** `/health` - Health check
- **GET** `/docs` - Interactive API documentation

### Usage Example

```python
import requests
import base64

# Encode your audio file
with open('audio.wav', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# Make API request
response = requests.post(
    'https://rajivdey-ai-voice-detection.hf.space/detect',
    headers={'x-api-key': 'hackathon_demo_key_2024'},
    json={
        'audio_base64': audio_base64,
        'language': 'en',
        'audio_format': 'wav'
    }
)

print(response.json())
```

### cURL Example

```bash
curl -X POST "https://rajivdey-ai-voice-detection.hf.space/detect" \
  -H "Content-Type: application/json" \
  -H "x-api-key: hackathon_demo_key_2024" \
  -d '{
    "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
    "language": "en",
    "audio_format": "wav"
  }'
```

## 📋 Request Format

### Required Fields

- `audio_base64` (string): Base64-encoded audio file (WAV or MP3)

### Optional Fields

- `language` (string): Language code - `en`, `ta`, `hi`, `ml`, `te` (default: auto-detect)
- `audio_format` (string): Audio format - `wav` or `mp3` (default: `wav`)

### Alternative Field Names Supported

The API is flexible and accepts various field name formats:

| Standard | Alternatives |
|----------|--------------|
| `audio_base64` | `audio`, `base64`, `audio_data` |
| `language` | `lang`, `language_code`, `ln` |
| `audio_format` | `format`, `file_format`, `type` |

### Example Request Body

```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "language": "en",
  "audio_format": "wav"
}
```

**Note:** Field order doesn't matter!

## 📤 Response Format

### Success Response (200 OK)

```json
{
  "classification": "human",
  "confidence": 0.95,
  "ai_probability": 0.05,
  "human_probability": 0.95,
  "explanation": "The model is very confident that this voice is human. Natural voice characteristics detected including organic pitch variation and typical human speech patterns.",
  "language": "en"
}
```

### Error Responses

**400 Bad Request** - Invalid audio data
```json
{
  "detail": "Invalid audio data: Error decoding audio"
}
```

**401 Unauthorized** - Missing API key
```json
{
  "detail": "API key required. Include 'x-api-key' header in your request."
}
```

**403 Forbidden** - Invalid API key
```json
{
  "detail": "Invalid API key"
}
```

## 🌍 Supported Languages

| Language | Code | Example |
|----------|------|---------|
| English | `en` | General American/British |
| Tamil | `ta` | தமிழ் |
| Hindi | `hi` | हिन्दी |
| Malayalam | `ml` | മലയാളം |
| Telugu | `te` | తెలుగు |

## 🔑 Authentication

Include your API key in the request header:

```
x-api-key: hackathon_demo_key_2024
```

## 🧠 Model Details

### Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: 98 acoustic features
  - 40 MFCCs (Mel-Frequency Cepstral Coefficients)
  - 8 Spectral features (centroid, rolloff, bandwidth, zero-crossing)
  - 3 Pitch features (mean, std, range)
  - 3 Energy features (RMS, energy)
  - 4 Temporal features (onset strength, tempogram)

### Performance
- **Training Accuracy**: 95%+
- **Cross-validation Score**: 85%+
- **Inference Time**: < 2 seconds per sample

### Detection Method

**AI-Generated Voices:**
- More regular spectral patterns
- Unnatural prosody and rhythm
- Consistent pitch with less variation
- Lower noise floor
- Repetitive patterns

**Human Voices:**
- Natural pitch variation
- Organic breath sounds
- Irregular speech patterns
- Environmental noise
- Dynamic amplitude changes

## 📊 Example Usage Scenarios

### Python SDK

```python
import requests
import base64

class VoiceDetectionClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
    
    def detect(self, audio_file_path, language='en'):
        with open(audio_file_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            f'{self.base_url}/detect',
            headers={'x-api-key': self.api_key},
            json={
                'audio_base64': audio_base64,
                'language': language,
                'audio_format': 'wav'
            }
        )
        
        return response.json()

# Usage
client = VoiceDetectionClient(
    api_key='hackathon_demo_key_2024',
    base_url='https://huggingface.co/spaces/rajivdey/AI-Voice-Detection/detect'
)

result = client.detect('audio.wav', language='en')
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript/Node.js

```javascript
const fs = require('fs');
const axios = require('axios');

async function detectVoice(audioFilePath, language = 'en') {
    const audioBuffer = fs.readFileSync(audioFilePath);
    const audioBase64 = audioBuffer.toString('base64');
    
    const response = await axios.post(
        'https://rajivdey-ai-voice-detection.hf.space/detect',
        {
            audio_base64: audioBase64,
            language: language,
            audio_format: 'wav'
        },
        {
            headers: {
                'x-api-key': 'hackathon_demo_key_2024',
                'Content-Type': 'application/json'
            }
        }
    );
    
    return response.data;
}

// Usage
detectVoice('audio.wav', 'en')
    .then(result => {
        console.log(`Classification: ${result.classification}`);
        console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
    });
```

## 🔧 Technical Specifications

### Audio Requirements
- **Formats**: WAV, MP3
- **Sample Rate**: Automatically resampled to 16kHz
- **Duration**: Maximum 5 seconds (auto-trimmed)
- **Channels**: Mono or Stereo (converted to mono)
- **File Size**: Maximum 5MB

### API Limits
- **Rate Limit**: 100 requests per minute
- **Timeout**: 30 seconds per request
- **Max Concurrent**: 10 requests

## 🛠️ Development

### Local Setup

```bash
# Clone repository
git clone https://github.com/rajivdey2/AI-Generated-Voice-Detection
cd ai-voice-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn src.api:app --reload --port 8000
```

### Docker

```bash
# Build image
docker build -t voice-detection .

# Run container
docker run -p 8000:8000 \
  -e API_KEY=hackathon_demo_key_2024 \
  -e ENVIRONMENT=production \
  voice-detection
```

## 📚 API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## 🤝 Contributing

This is a hackathon project. For issues or questions, please open an issue on the repository.

## 📄 License

MIT License - See LICENSE file for details

## 🏆 Hackathon Submission

This API was developed for [GUVI Hackathon] to detect AI-generated voices across multiple Indian languages.

### Team
- [Rajiv Dey]
- [Sai ,Harsh ,Subhasmita ]

### Project Links
- **Live API**: https:// https://rajivdey-ai-voice-detection.hf.space
- **Documentation**: https://huggingface.co/spaces/rajivdey/AI-Voice-Detection
- **GitHub**: https://github.com/rajivdey2/AI-Generated-Voice-Detection

---


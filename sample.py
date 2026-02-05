import base64
import json

# Encode audio
with open('data/raw/human/hindi_human_001.wav', 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

# Create JSON payload
payload = {
    "audio_base64": audio_base64,
    "language": "english"
}

# Save to file or print
print(json.dumps(payload, indent=2))
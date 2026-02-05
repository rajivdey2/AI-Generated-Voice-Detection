"""
Proper test script for local API testing
Handles Base64 encoding correctly
"""

import requests
import base64
import json
import sys

def test_api_with_file(audio_file, language='en'):
    """Test API with proper Base64 encoding"""
    
    print("=" * 60)
    print("Testing Local API")
    print("=" * 60)
    
    # Read and encode audio file
    print(f"\n1. Reading audio file: {audio_file}")
    try:
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        # Encode to base64 - CLEAN encoding
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"   ✓ File read successfully ({len(audio_bytes)} bytes)")
        print(f"   ✓ Base64 encoded ({len(audio_base64)} characters)")
        
        # Verify encoding (decode and check)
        test_decode = base64.b64decode(audio_base64)
        if test_decode == audio_bytes:
            print(f"   ✓ Base64 encoding verified")
        else:
            print(f"   ✗ Base64 encoding verification failed!")
            return
            
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
        return
    
    # Prepare request
    url = "http://localhost:8000/detect"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "hackathon_demo_key_2024"
    }
    
    payload = {
        "audio_base64": audio_base64,
        "audio_format": "wav",
        "language": language
        
    }
    
    # Make request
    print(f"\n2. Making API request to: {url}")
    print(f"   Language: {language}")
    print(f"   Payload size: {len(json.dumps(payload))} bytes")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"\n3. Response received")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 60)
            print("✅ SUCCESS - API Response:")
            print("=" * 60)
            print(json.dumps(result, indent=2))
            print("=" * 60)
            
            print(f"\n📊 Summary:")
            print(f"   Classification: {result['classification']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Human Probability: {result['human_probability']:.2%}")
            print(f"   AI Probability: {result['ai_probability']:.2%}")
            
        else:
            print("\n" + "=" * 60)
            print(f"❌ ERROR - Status {response.status_code}")
            print("=" * 60)
            print(response.text)
            
            # Try to parse error
            try:
                error_data = response.json()
                print("\nError Details:")
                print(json.dumps(error_data, indent=2))
            except:
                pass
                
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("   Make sure the API is running:")
        print("   uvicorn src.api:app --reload")
        
    except requests.exceptions.Timeout:
        print("\n❌ ERROR: Request timed out (>30s)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

def test_all_languages():
    """Test all supported languages"""
    
    languages = {
        'en': 'english',
        'ta': 'tamil',
        'hi': 'hindi',
        'ml': 'malayalam',
        'te': 'telugu'
    }
    
    print("\n" + "=" * 60)
    print("Testing All Languages")
    print("=" * 60)
    
    for lang_code, lang_name in languages.items():
        print(f"\n{'='*60}")
        print(f"Testing {lang_name.upper()} ({lang_code})")
        print('='*60)
        
        # Test human voice
        human_file = f"data/raw/human/{lang_name}_human_001.wav"
        print(f"\n🧑 Testing HUMAN voice:")
        test_api_with_file(human_file, lang_code)
        
        # Test AI voice
        print(f"\n🤖 Testing AI voice:")
        ai_file = f"data/raw/ai/{lang_name}_ai_001.wav"
        test_api_with_file(ai_file, lang_code)

def save_base64_to_file(audio_file, output_file="base64_payload.json"):
    """Save properly encoded Base64 to a file for manual testing"""
    
    print(f"\n📝 Saving Base64 payload to: {output_file}")
    
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    payload = {
        "audio_base64": audio_base64,
        "language": "en",
        "audio_format": "wav"
    }
    
    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"   ✓ Saved! You can use this in Swagger UI")
    print(f"   File: {output_file}")

if __name__ == "__main__":
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        print("✓ API is running")
        print(f"  Status: {health.json().get('status')}")
    except:
        print("\n❌ API is NOT running!")
        print("Please start the API first:")
        print("  uvicorn src.api:app --reload\n")
        sys.exit(1)
    
    # Test based on arguments
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else 'en'
        
        if audio_file == "--all":
            test_all_languages()
        elif audio_file == "--save":
            file_to_encode = sys.argv[2] if len(sys.argv) > 2 else "data/raw/human/english_human_001.wav"
            save_base64_to_file(file_to_encode)
        else:
            test_api_with_file(audio_file, language)
    else:
        # Default test
        print("\nUsage:")
        print("  python test_api_local.py <audio_file> [language]")
        print("  python test_api_local.py --all  (test all languages)")
        print("  python test_api_local.py --save <audio_file>  (save Base64 to file)")
        print("\nRunning default test...\n")
        
        default_file = "data/raw/human/english_human_001.wav"
        test_api_with_file(default_file, 'en')
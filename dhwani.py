import sys
import io
import dwani
import os
from deep_translator import GoogleTranslator
import hashlib
from datetime import datetime

# Ensure UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Dhwani API configuration
dwani.api_key = os.getenv("DWANI_API_KEY", "preran248@gmail.com_dwani")  
dwani.api_base = os.getenv("DWANI_API_BASE_URL", "https://dwani-dwani-api.hf.space")

# Cache for translations
translation_cache = {}

def translate_to_kannada(text):
    # Check cache first
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    if text_hash in translation_cache:
        return translation_cache[text_hash]
    
    try:
        translator = GoogleTranslator(source='en', target='kn')
        result = translator.translate(text)
        translation_cache[text_hash] = result
        return result
    except Exception as e:
        return f"Error during translation: {str(e)}"

def text_to_speech(kannada_text):
    try:
        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_hash = hashlib.md5(kannada_text.encode('utf-8')).hexdigest()[:8]
        output_file = f"output_{timestamp}_{text_hash}.mp3"
        
        # Generate speech using Dhwani TTS
        audio_data = dwani.Audio.speech(
            input=kannada_text,
            response_format="mp3"
        )
        if not audio_data:
            return "Error during TTS: No audio data returned"
        
        # Save audio
        with open(output_file, 'wb') as file:
            file.write(audio_data)
        return output_file
    except Exception as e:
        return f"Error during TTS: {str(e)}"

def main():
    print("English to Kannada Translator with Dhwani TTS")
    print("Enter 'quit' to exit")
    
    while True:
        english_text = input("Enter English text to translate: ")
        if english_text.lower() == 'quit':
            break
        if not english_text.strip():
            print("Please enter some text.")
            continue
            
        # Translate to Kannada
        kannada_text = translate_to_kannada(english_text)
        if kannada_text.startswith("Error"):
            print(kannada_text)
            continue
            
        # Generate speech
        audio_file = text_to_speech(kannada_text)
        if audio_file.startswith("Error"):
            print(audio_file)
            continue
            
        print(f"Kannada translation: {kannada_text}")
        print(f"Audio saved as: {audio_file}\n")

if __name__ == "__main__":
    main()
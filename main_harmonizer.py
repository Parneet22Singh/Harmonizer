import os
from modules.transcriber import transcribe_audio
from elevenlabs import ElevenLabs, play
import playsound
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "Your_API_KEY"))

# Initialize ElevenLabs client
API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOU_API_KEY")
client = ElevenLabs(api_key=API_KEY)

# Voice options by language/accent
VOICE_IDS = {
    "1": ("US English (Male)", "EXAVITQu4vr4xnSDxMaL"),
    "2": ("British English (Female)", "21m00Tcm4TlvDq8ikWAM"),
    "3": ("Australian English (Female)", "AZnzlk1XvdvUeBnXmlld"),
    "4": ("Spanish (Female)", "pNInz6obpgDQGcFmaJgB"),
    "5": ("French (Female)", "EXAVITQu4vr4xnSDxMaL"),
    "6": ("Indian English (Male)", "m5qndnI7u4OAdXhH0Mr5"),
    "7": ("Japanese Male (Otani)", "3JDquces8E8bkmvbh6Bc"),
    "8": ("Indian Female (Anika)","RXe6OFmxoC0nlSWpuCDy")
}

LANGUAGE_CODES = {
    "1": "en",
    "2": "es",
    "3": "fr",
    "4": "hi",
    "5": "ja",
}

LANGUAGE_NAMES = {
    "1": "English",
    "2": "Spanish",
    "3": "French",
    "4": "Hindi",
    "5": "Japanese",
}

def adapt_text(text: str, language: str, tone: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        f"Translate and rewrite the following English sentence into {language} using a {tone} tone. Preserve the core message, make sure that only the translation is generated nothing else and make it sound natural for a native {language} speaker:\n\n"
        f"{text}"
    )

    try:
        print("\n🔄 Sending prompt to Gemini...")
        response = model.generate_content(prompt)

        if response and response.candidates:
            parts = response.candidates[0].content.parts
            final_text = ''.join(part.text for part in parts if hasattr(part, "text"))
            print("✅ Adapted response received.\n")
            return final_text.strip()

        return "⚠️ Gemini returned no usable content."

    except Exception as e:
        return f"❌ Error from Gemini: {e}"

def generate_and_play(text: str, voice_id: str, output_path="output.mp3"):
    print(f"\n🎤 Generating audio with voice ID {voice_id}...")
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",  # Enables better accent/language handling
        output_format="mp3_44100"
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print(f"✅ Audio saved to {output_path}")

    playsound.playsound(output_path)

def main():
    input_path = "audio_samples/accented_input1.mp3"
    if not os.path.isfile(input_path):
        print(f"❌ File not found at {input_path}. Exiting.")
        return

    print("\nChoose target language:")
    for key, lang in LANGUAGE_NAMES.items():
        print(f"{key}. {lang}")
    lang_choice = input("Enter choice number: ").strip()
    language = LANGUAGE_NAMES.get(lang_choice, "English")
    language_code = LANGUAGE_CODES.get(lang_choice, "en")

    print("\nChoose tone:")
    tones = ["formal", "casual", "neutral"]
    for i, tone in enumerate(tones, 1):
        print(f"{i}. {tone}")
    tone_choice = input("Enter choice number: ").strip()
    tone = tones[int(tone_choice) - 1] if tone_choice.isdigit() and 1 <= int(tone_choice) <= len(tones) else "neutral"

    print("\nChoose voice option:")
    for key, (desc, _) in VOICE_IDS.items():
        print(f"{key}. {desc}")
    voice_choice = input("Enter choice number: ").strip()
    voice_id = VOICE_IDS.get(voice_choice, VOICE_IDS["1"])[1]

    print("\n🔤 Transcribing audio...")
    original_text = transcribe_audio(input_path, language_code)
    print(f"\n🗣️ Original Transcript:\n{original_text}")

    adapted_text = adapt_text(original_text, language, tone)
    print(f"\n📝 Adapted Transcript:\n{adapted_text}")

    generate_and_play(adapted_text, voice_id)

if __name__ == "__main__":
    main()

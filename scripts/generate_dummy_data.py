import os
import pyttsx3
import soundfile as sf
import numpy as np

def synthesize_voice(text, voice_id, out_path, sr=22050):
    """
    Generate speech using pyttsx3 and save to .wav file.
    """
    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', 160)  # slow-ish, more musical
    engine.save_to_file(text, out_path)
    engine.runAndWait()

def get_voices_by_type():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    type1_voices = [v.id for v in voices if 'type1' in v.name.lower()]
    type2_voices = [v.id for v in voices if 'type2' in v.name.lower()]
    return type1_voices, type2_voices

def generate_samples(out_root="data/raw", n_samples=5):
    os.makedirs(f"{out_root}/type1", exist_ok=True)
    os.makedirs(f"{out_root}/type2", exist_ok=True)

    texts = [
        "The sun sets over the city skyline.",
        "I dream of singing under the stars.",
        "Voices carry across the ocean breeze.",
        "Let the music be your guide.",
        "Echoes of harmony and light."
    ]

    type1_voices, type2_voices = get_voices_by_type()

    if not type1_voices or not type2_voices:
        raise RuntimeError("Could not find both type1 and type2 voices on this system.")

    print("[🎙️] Generating type1 voice samples...")
    for i, text in enumerate(texts[:n_samples]):
        synthesize_voice(text, type1_voices[0], f"{out_root}/type1/type1_{i}.wav")

    print("[🎙️] Generating type2 voice samples...")
    for i, text in enumerate(texts[:n_samples]):
        synthesize_voice(text, type2_voices[0], f"{out_root}/type2/type2_{i}.wav")

    print(f"[✓] Generated {n_samples * 2} audio files in '{out_root}'")

if __name__ == "__main__":
    generate_samples()

import os
import pyttsx3

def synthesize_voice(text, voice_id, out_path):
    """
    Use pyttsx3 to synthesize and save audio.
    """
    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', 160)
    engine.save_to_file(text, out_path)
    engine.runAndWait()

def get_voices_by_type():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    type1_voices = [v.id for v in voices if 'type1' in v.name.lower()]
    type2_voices = [v.id for v in voices if 'type2' in v.name.lower()]
    return type1_voices, type2_voices

def generate_paired_samples(covers_dir="data/fine_tune/my_covers", targets_dir="data/fine_tune/targets", n_samples=5):
    os.makedirs(covers_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)

    texts = [
        "The moonlight guides my voice.",
        "Sing to the rhythm of your heart.",
        "Melodies rise and fall with grace.",
        "Voices intertwined in harmony.",
        "Music is the soul of emotion."
    ]

    type1_voices, type2_voices = get_voices_by_type()

    if not type1_voices or not type2_voices:
        raise RuntimeError("Both type1 and type2 voices are required.")

    print("[🎙️] Generating paired samples...")
    for i, text in enumerate(texts[:n_samples]):
        cover_path = os.path.join(covers_dir, f"cover_{i}.wav")
        target_path = os.path.join(targets_dir, f"target_{i}.wav")
        synthesize_voice(text, type1_voices[0], cover_path)
        synthesize_voice(text, type2_voices[0], target_path)

    print(f"[✓] Generated {n_samples} paired samples.")

if __name__ == "__main__":
    generate_paired_samples()

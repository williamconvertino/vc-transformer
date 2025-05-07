import os
import librosa
import pandas as pd
from tqdm import tqdm
from datasets.audio_utils import load_audio, wav_to_mel
import numpy as np

def preprocess_dataset(input_dir: str, output_dir: str, sr: int = 22050):
    """
    Preprocess all .wav files in input_dir to mel spectrograms and save them.

    Args:
        input_dir: Path to raw audio
        output_dir: Path to store mel .npy files
    """
    os.makedirs(output_dir, exist_ok=True)
    for speaker in os.listdir(input_dir):
        speaker_dir = os.path.join(input_dir, speaker)
        out_dir = os.path.join(output_dir, speaker)
        os.makedirs(out_dir, exist_ok=True)

        for fname in tqdm(os.listdir(speaker_dir), desc=f"Processing {speaker}"):
            if not fname.endswith('.wav'):
                continue
            wav = load_audio(os.path.join(speaker_dir, fname), sr)
            mel = wav_to_mel(wav, sr)
            np.save(os.path.join(out_dir, fname.replace('.wav', '.npy')), mel)

def generate_metadata(processed_dir: str, metadata_path: str, split_ratio=(0.8, 0.1, 0.1)):
    """
    Generate a metadata CSV for dataset usage.

    Columns: file_path, speaker, type, split

    Args:
        processed_dir: Root of processed mel files
        metadata_path: Path to save CSV
        split_ratio: Train/val/test split
    """
    rows = []
    for speaker in os.listdir(processed_dir):
        speaker_dir = os.path.join(processed_dir, speaker)
        files = [f for f in os.listdir(speaker_dir) if f.endswith('.npy')]
        type = 0 if speaker.lower() == "alt" else 1  # Adjust based on your structure

        num_files = len(files)
        train_end = int(split_ratio[0] * num_files)
        val_end = int((split_ratio[0] + split_ratio[1]) * num_files)

        for i, f in enumerate(sorted(files)):
            split = 'train' if i < train_end else 'val' if i < val_end else 'test'
            rows.append({
                'file_path': os.path.join(speaker_dir, f),
                'speaker': speaker,
                'type': type,
                'split': split
            })

    pd.DataFrame(rows).to_csv(metadata_path, index=False)

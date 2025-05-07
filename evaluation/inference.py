import torch
import torchaudio
from models.transformer import VoiceTransformer
from models.vocoder import Vocoder
from datasets.audio_utils import load_audio, wav_to_mel, mel_to_tensor
import numpy as np
import soundfile as sf

def infer(model: VoiceTransformer, vocoder: Vocoder, input_wav_path: str, output_path: str, sr: int = 22050):
    """
    Run inference on a single waveform and save the result.

    Args:
        model: Trained VoiceTransformer
        vocoder: Trained vocoder (e.g., HiFi-GAN)
        input_wav_path: Path to input .wav file
        output_path: Where to save the generated .wav
        sr: Sampling rate
    """
    model.eval()
    wav = load_audio(input_wav_path, sr=sr)
    mel = wav_to_mel(wav, sr=sr)
    mel_tensor = mel_to_tensor(mel).unsqueeze(0).to(model.model.device)  # [1, T, F]

    with torch.no_grad():
        output = model(mel_tensor)
        mel_out = output["reconstruction"].transpose(1, 2)  # [1, F, T]

    audio_out = vocoder.mel_to_audio(mel_out)
    sf.write(output_path, audio_out, sr)
    print(f"Inference complete. Output saved to {output_path}")

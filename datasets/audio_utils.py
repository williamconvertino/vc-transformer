import librosa
import numpy as np
import torch

def load_audio(path: str, sr: int = 22050) -> np.ndarray:
    """
    Load an audio file.

    Args:
        path: Path to .wav file
        sr: Sampling rate

    Returns:
        Numpy array of waveform
    """
    wav, _ = librosa.load(path, sr=sr)
    return wav

def wav_to_mel(wav: np.ndarray, sr: int = 22050, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80) -> np.ndarray:
    """
    Convert waveform to mel-spectrogram.

    Returns:
        Mel spectrogram [n_mels, time]
    """
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def mel_to_tensor(mel: np.ndarray) -> torch.Tensor:
    """
    Convert mel-spectrogram (numpy) to tensor [T, F].

    Returns:
        torch.Tensor of shape [T, F]
    """
    mel = mel.T  # [T, F]
    return torch.tensor(mel, dtype=torch.float32)

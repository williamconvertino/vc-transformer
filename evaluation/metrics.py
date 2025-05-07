import torch
import torchaudio
import numpy as np

def compare_mel_similarity(mel1: torch.Tensor, mel2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two mel-spectrograms.

    Args:
        mel1, mel2: [T, F] tensors

    Returns:
        Cosine similarity score (float)
    """
    mel1_flat = mel1.view(-1)
    mel2_flat = mel2.view(-1)
    return torch.nn.functional.cosine_similarity(mel1_flat, mel2_flat, dim=0).item()

def compute_pitch_accuracy(source_wav: np.ndarray, target_wav: np.ndarray, sr: int = 22050) -> float:
    """
    Compute pitch contour correlation between two waveforms using YIN.

    Args:
        source_wav, target_wav: Numpy arrays
        sr: Sampling rate

    Returns:
        Correlation coefficient
    """
    pitch_source, _ = torchaudio.functional.detect_pitch_frequency(torch.tensor(source_wav).unsqueeze(0), sample_rate=sr)
    pitch_target, _ = torchaudio.functional.detect_pitch_frequency(torch.tensor(target_wav).unsqueeze(0), sample_rate=sr)

    if pitch_source.numel() == 0 or pitch_target.numel() == 0:
        return 0.0

    # Match lengths
    min_len = min(pitch_source.shape[-1], pitch_target.shape[-1])
    pitch_source = pitch_source[..., :min_len]
    pitch_target = pitch_target[..., :min_len]

    corr = np.corrcoef(pitch_source.squeeze().numpy(), pitch_target.squeeze().numpy())[0, 1]
    return float(corr)

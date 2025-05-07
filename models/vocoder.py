import torch
import numpy as np
from models.hifigan import Generator  # Assume HiFi-GAN generator is placed in models/hifigan
import torchaudio

class Vocoder:
    """
    Wrapper for HiFi-GAN vocoder.
    """

    def __init__(self, checkpoint_path: str):
        self.model = Generator()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["generator"])
        self.model.eval()
        self.model.remove_weight_norm()

    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convert mel-spectrogram to waveform.

        Args:
            mel: [1, 80, T] tensor

        Returns:
            Numpy waveform
        """
        with torch.no_grad():
            audio = self.model(mel).squeeze().cpu().numpy()
        return audio

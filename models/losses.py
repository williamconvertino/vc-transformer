import torch
import torch.nn.functional as F

def masked_mel_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked L1 loss for mel spectrogram prediction.

    Args:
        pred: [B, T, F] predicted mel
        target: [B, T, F] ground truth
        mask: [B, T, 1] binary mask

    Returns:
        Loss value
    """
    return F.l1_loss(pred * mask, target * mask)

def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary or categorical cross-entropy loss for type classification.

    Args:
        logits: [B, 2]
        labels: [B] (0 or 1)

    Returns:
        Loss value
    """
    return F.cross_entropy(logits, labels)

def mel_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss over full mel spectrograms.

    Args:
        pred: [B, T, F]
        target: [B, T, F]

    Returns:
        Scalar loss
    """
    return F.l1_loss(pred, target)

def perceptual_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for perceptual loss — to be filled in with pretrained model (e.g., wav2vec2 or CREPE).

    Args:
        x, y: Mel or waveform tensors

    Returns:
        Scalar loss
    """
    # Placeholder implementation
    return F.mse_loss(x, y)

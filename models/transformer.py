import torch
import torch.nn as nn
from transformers import SpeechT5Model, SpeechT5Config

class VoiceTransformer(nn.Module):
    """
    Transformer model wrapper for voice conversion using SpeechT5.
    Supports both pretraining (mask prediction, classification) and fine-tuning.
    """

    def __init__(self, pretrained_model_name: str = "microsoft/speecht5_asr", mode: str = "pretrain"):
        super().__init__()
        self.mode = mode
        self.model = SpeechT5Model.from_pretrained(pretrained_model_name)

        # Optional classification head (for type)
        if mode == "pretrain":
            self.cls_head = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 2)  # Type1 / Type2
            )

        # Optional projection head for output mel predictions
        if mode == "pretrain" or mode == "finetune":
            self.mel_proj = nn.Linear(self.model.config.hidden_size, 80)  # Mel bins

    def forward(self, mel: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            mel: [B, T, F] tensor (mel-spectrogram)
            attention_mask: Optional attention mask [B, T]

        Returns:
            Dictionary with:
                - 'reconstruction': mel prediction
                - 'cls_logits': type prediction (if in pretrain)
                - 'hidden_states': raw hidden states
        """
        inputs_embeds = mel  # [B, T, 80]

        # Run through transformer
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output.last_hidden_state  # [B, T, H]

        out = {
            "hidden_states": hidden_states,
        }

        if hasattr(self, 'mel_proj'):
            out['reconstruction'] = self.mel_proj(hidden_states)

        if hasattr(self, 'cls_head'):
            cls_token = hidden_states[:, 0, :]  # Use first token as CLS
            out['cls_logits'] = self.cls_head(cls_token)

        return out

    def load_pretrained_weights(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint['model_state_dict'])

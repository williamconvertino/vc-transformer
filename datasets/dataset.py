import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class VoiceDataset(Dataset):
    """
    PyTorch dataset for voice data.

    Modes:
    - "pretrain": Only input mel + type
    - "finetune": Paired samples (input mel, target mel)
    """

    def __init__(self, metadata_csv: str, mode: str = "pretrain"):
        self.data = pd.read_csv(metadata_csv)
        self.mode = mode

        if mode not in ["pretrain", "finetune"]:
            raise ValueError("Invalid mode for dataset")

        if mode == "finetune":
            # Assumes an additional column `target_path` exists
            if "target_path" not in self.data.columns:
                raise ValueError("Missing `target_path` column for finetune mode")

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        mel = np.load(row['file_path'])
        mel_tensor = torch.tensor(mel.T, dtype=torch.float32)  # [T, F]

        sample = {
            "mel": mel_tensor,
            "type": torch.tensor(row["type"], dtype=torch.long),
        }

        if self.mode == "finetune":
            target_mel = np.load(row["target_path"])
            target_tensor = torch.tensor(target_mel.T, dtype=torch.float32)
            sample["target_mel"] = target_tensor

        return sample

    def __len__(self):
        return len(self.data)

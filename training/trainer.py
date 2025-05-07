import torch
from torch.utils.data import DataLoader
import os
from models.losses import masked_mel_loss, classification_loss, mel_reconstruction_loss

class Trainer:
    """
    Shared trainer for both pretraining and fine-tuning.
    """

    def __init__(self, config, model, dataloader, val_dataloader, optimizer, scheduler=None, mode="pretrain"):
        self.config = config
        self.model = model.to(config["device"])
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.device = config["device"]

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.dataloader:
            mel = batch["mel"].to(self.device)  # [B, T, F]
            output = self.model(mel)

            if self.mode == "pretrain":
                loss1 = masked_mel_loss(output["reconstruction"], mel, mask=torch.ones_like(mel))
                loss2 = classification_loss(output["cls_logits"], batch["type"].to(self.device))
                loss = loss1 + self.config.get("cls_weight", 1.0) * loss2

            elif self.mode == "finetune":
                target = batch["target_mel"].to(self.device)
                loss = mel_reconstruction_loss(output["reconstruction"], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                mel = batch["mel"].to(self.device)
                output = self.model(mel)

                if self.mode == "pretrain":
                    loss1 = masked_mel_loss(output["reconstruction"], mel, mask=torch.ones_like(mel))
                    loss2 = classification_loss(output["cls_logits"], batch["type"].to(self.device))
                    loss = loss1 + self.config.get("cls_weight", 1.0) * loss2

                elif self.mode == "finetune":
                    target = batch["target_mel"].to(self.device)
                    loss = mel_reconstruction_loss(output["reconstruction"], target)

                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def save_checkpoint(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

import torch
import yaml
from torch.utils.data import DataLoader
from models.transformer import VoiceTransformer
from datasets.dataset import VoiceDataset
from training.trainer import Trainer
from torch.optim import AdamW

def run_pretraining(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = VoiceTransformer(mode="pretrain")
    dataset = VoiceDataset(metadata_csv=config["dataset"]["metadata_csv"], mode="pretrain")

    train_set = [x for x in dataset.data.itertuples() if x.split == "train"]
    val_set = [x for x in dataset.data.itertuples() if x.split == "val"]

    train_dataset = VoiceDataset(metadata_csv=config["dataset"]["metadata_csv"], mode="pretrain")
    val_dataset = VoiceDataset(metadata_csv=config["dataset"]["metadata_csv"], mode="pretrain")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    trainer = Trainer(
        config=config,
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        mode="pretrain"
    )

    for epoch in range(config["epochs"]):
        train_loss = trainer.train_one_epoch()
        val_loss = trainer.validate()
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if (epoch + 1) % config.get("save_every", 5) == 0:
            trainer.save_checkpoint(f"{config['checkpoint_dir']}/pretrain_epoch{epoch+1}.pt")

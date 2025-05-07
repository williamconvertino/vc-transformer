import argparse
import os
from datasets.preprocess import preprocess_dataset, generate_metadata
import pandas as pd
from glob import glob

def make_finetune_csv(covers_dir, targets_dir, output_csv, split_ratio=(0.8, 0.1, 0.1)):
    """
    Create a metadata CSV for paired training (fine-tuning).

    Assumes files in covers_dir and targets_dir have matching names.
    """

    cover_files = sorted(glob(os.path.join(covers_dir, '*.npy')))
    target_files = sorted(glob(os.path.join(targets_dir, '*.npy')))
    assert len(cover_files) == len(target_files), "Mismatch in cover/target files"

    rows = []
    num_files = len(cover_files)
    train_end = int(split_ratio[0] * num_files)
    val_end = int((split_ratio[0] + split_ratio[1]) * num_files)

    for i, (cover_path, target_path) in enumerate(zip(cover_files, target_files)):
        split = 'train' if i < train_end else 'val' if i < val_end else 'test'
        rows.append({
            'file_path': cover_path,
            'target_path': target_path,
            'speaker': 'type1',
            'type': 0,
            'split': split
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[✓] Fine-tune metadata saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw", help="Path to raw audio (type 1/type 2 subfolders)")
    parser.add_argument("--processed_dir", default="data/processed", help="Where to save mel-spectrograms")
    parser.add_argument("--metadata_dir", default="data/metadata", help="Where to save CSVs")
    parser.add_argument("--covers_dir", help="Path to .wav files of your covers (optional)")
    parser.add_argument("--targets_dir", help="Path to .wav files of matching type 2 vocals (optional)")
    args = parser.parse_args()

    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.metadata_dir, exist_ok=True)

    print("🔄 Preprocessing pretraining data...")
    preprocess_dataset(args.raw_dir, args.processed_dir, save_cleaned_wav=True)

    print("📝 Creating pretraining metadata...")
    pretrain_csv = os.path.join(args.metadata_dir, "pretrain_metadata.csv")
    generate_metadata(args.processed_dir, pretrain_csv)

    if args.covers_dir and args.targets_dir:
        print("🎵 Preprocessing fine-tune data...")
        cover_out = os.path.join(args.processed_dir, "covers")
        target_out = os.path.join(args.processed_dir, "targets")
        os.makedirs(cover_out, exist_ok=True)
        os.makedirs(target_out, exist_ok=True)

        preprocess_dataset(args.covers_dir, cover_out, save_cleaned_wav=True)
        preprocess_dataset(args.targets_dir, target_out, save_cleaned_wav=True)

        print("📝 Creating fine-tuning metadata...")
        finetune_csv = os.path.join(args.metadata_dir, "finetune_metadata.csv")
        make_finetune_csv(cover_out, target_out, finetune_csv)

if __name__ == "__main__":
    main()

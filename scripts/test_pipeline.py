"""
Quick script to test that the full model loads and processes an audio file end-to-end.
"""

from models.transformer import VoiceTransformer
from models.vocoder import Vocoder
from evaluation.inference import infer
import yaml

def main():
    with open("config/inference.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("[INFO] Loading model...")
    model = VoiceTransformer(mode='finetune')
    model.load_pretrained_weights(cfg['model_checkpoint'])

    print("[INFO] Loading vocoder...")
    vocoder = Vocoder(cfg['vocoder_checkpoint'])

    print("[INFO] Running inference...")
    infer(model, vocoder, cfg['input_wav'], cfg['output_wav'], sr=cfg['sr'])

if __name__ == "__main__":
    main()

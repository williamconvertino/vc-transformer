#!/bin/bash
python -c "
from evaluation.inference import infer
from models.transformer import VoiceTransformer
from models.vocoder import Vocoder
import yaml

with open('config/inference.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

model = VoiceTransformer(mode='finetune')
model.load_pretrained_weights(cfg['model_checkpoint'])
vocoder = Vocoder(cfg['vocoder_checkpoint'])

infer(model, vocoder, cfg['input_wav'], cfg['output_wav'], sr=cfg['sr'])
"

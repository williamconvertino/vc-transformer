#!/bin/bash

# Create checkpoint directory
mkdir -p checkpoints/vocoder

echo "Downloading pretrained HiFi-GAN vocoder..."

# Download universal vocoder (trained on VCTK) from official repo
wget -O checkpoints/vocoder/generator.pth "https://github.com/jik876/hifi-gan/releases/download/v0.1/universal_generator.pth.tar"

# Clone HiFi-GAN repo if needed
if [ ! -d "models/hifigan" ]; then
    echo "Cloning HiFi-GAN model code..."
    git clone https://github.com/jik876/hifi-gan.git temp_hifigan
    cp -r temp_hifigan/models models/hifigan
    rm -rf temp_hifigan
fi

echo "✅ Vocoder setup complete."

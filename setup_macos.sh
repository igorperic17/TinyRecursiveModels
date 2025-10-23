#!/bin/bash

# macOS setup script for TinyRecursiveModels
# This script sets up the environment with MLX dependencies instead of triton

echo "🍎 Setting up TinyRecursiveModels for macOS..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This script is optimized for Apple Silicon (M1/M2/M3). Intel Macs may have compatibility issues."
fi

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install macOS-specific requirements
echo "📥 Installing macOS-specific dependencies..."
pip install -r requirements-macos.txt

# Test the installation
echo "🧪 Testing installation..."
python -c "
import torch
import mlx
from adam_atan2_macos import AdamATan2
print('✅ PyTorch version:', torch.__version__)
print('✅ MLX imported successfully')
print('✅ AdamATan2 macOS version imported successfully')
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ MPS available:', torch.backends.mps.is_available())
print('✅ All dependencies installed successfully!')
"

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   source .venv/bin/activate"
echo "   python train_language.py"
echo "   # or"
echo "   python setup_language_training.py --task conversation"

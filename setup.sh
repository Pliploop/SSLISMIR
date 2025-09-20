#!/bin/bash

# Music SSL ISMIR 2025 Setup Script
# This script creates a conda environment, installs dependencies, and downloads data

set -e  # Exit on any error

echo "🎵 Setting up Music SSL ISMIR 2025 environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment if it doesn't exist
if [ ! -d $(conda info --base)/envs/ismir_ssl_2025 ]; then
echo "📦 Creating conda environment 'ismir_ssl_2025'..."
conda create -n ismir_ssl_2025 python=3.10 -y
fi

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ismir_ssl_2025

# Install other requirements
echo "📚 Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Install additional dependencies for the download script
echo "📥 Installing additional dependencies..."
pip install mirdata pandas

# Create data directory
echo "📁 Creating data directory..."
mkdir -p data/giantsteps

# Run the download script
echo "⬇️  Downloading GiantSteps dataset..."
python scripts/download_giantsteps.py

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ismir_ssl_2025"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
echo ""
echo "🎉 Happy coding!"

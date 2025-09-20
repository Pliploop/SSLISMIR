# Appendix A: Practical Setup

This appendix provides essential instructions for setting up your environment and running the code examples in this tutorial.

## Quick Setup

The easiest way to get started is to use the provided setup script:

```bash
# Clone the repository
git clone https://github.com/yourusername/ssl-music-tutorial.git
cd ssl-music-tutorial

# Run the setup script
chmod +x setup.sh
./setup.sh
```

This script will:
- Create a conda environment with Python 3.10
- Install all required dependencies
- Download the GiantSteps dataset
- Set up the necessary directories

## Manual Setup

If you prefer manual setup or the script doesn't work on your system:

### 1. Environment Setup
```bash
# Create conda environment
conda create -n ssl_music python=3.10 -y
conda activate ssl_music

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
```bash
# Create data directory
mkdir -p data/giantsteps

# Download dataset (if available)
python scripts/download_giantsteps.py
```

## Building and Viewing the Book

### Local Development
```bash
# Build the book
jupyter book build book/

# Serve locally
jupyter book serve book/_build/html
```

The book will be available at `http://localhost:8000`

### Using the Build Script
```bash
# Build and serve in one command
./scripts/build_book.sh --serve
```

## Running the Notebooks

### Jupyter Lab
```bash
# Start Jupyter Lab
jupyter lab

# Navigate to the notebooks in the book/ directory
```

### Individual Notebooks
The hands-on labs are located in:
- `book/part2_jea/hands_on_contrastive.ipynb`
- `book/part3_masked_modeling/hands_on_masked_modeling.ipynb`
- `book/part4_equivariant_ssl/hands_on_equivariant.ipynb`
- `book/part5_generative_ssl/hands_on_generative.ipynb`

## Troubleshooting

### Common Issues
- **Conda not found**: Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
- **CUDA issues**: Ensure PyTorch CUDA version matches your CUDA installation
- **Memory issues**: Reduce batch size in the notebooks for systems with limited RAM

### Getting Help
- Check the GitHub issues page for common problems
- Refer to the individual notebook documentation
- Contact the authors for technical support

---

*Next: [Appendix B: Further Reading](further_reading.md)*

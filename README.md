# Music Self-Supervised Learning for ISMIR 2025

This repository contains the code and materials for a tutorial on **Joint Embedding Architectures for Music Self-Supervised Learning** presented at the 25th International Society for Music Information Retrieval Conference (ISMIR 2025). The tutorial focuses on contrastive learning approaches for learning music representations from audio data.

## 🎵 About This Tutorial

This tutorial explores state-of-the-art self-supervised learning techniques specifically designed for music audio data. We cover joint embedding architectures that learn meaningful representations by contrasting different views of the same musical content, enabling powerful music understanding without requiring large amounts of labeled data.

### Key Topics Covered:
- **Contrastive Learning for Music**: Learning representations by contrasting positive and negative pairs
- **Multi-View Augmentation**: Creating different views of the same musical content
- **Joint Embedding Architectures**: Building encoders that map audio to shared embedding spaces
- **Music-Specific Data Processing**: Mel spectrograms, audio augmentation, and temporal modeling
- **Evaluation and Visualization**: Assessing learned representations and understanding what the model learns

## 🚀 Quick Start

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Music-SSL-ISMIR.git
   cd Music-SSL-ISMIR
   ```

2. **Run the setup script:**
   ```bash setup.sh
   ```

   This will:
   - Create a conda environment named `ismir_ssl_2025`
   - Install all required dependencies
   - Download the GiantSteps dataset
   - Set up the project structure

3. **Activate the environment:**
   ```bash
   conda activate ismir_ssl_2025
   ```

```
Music-SSL-ISMIR/
├── src/
│   ├── models/
│   │   ├── backbones.py          # MLP and other backbone architectures
│   │   └── training_wrappers.py  # Training loops and model wrappers
│   ├── data/
│   │   ├── dataset.py            # Dataset classes and data loaders
│   │   └── collate.py            # Custom collate functions for batching
│   ├── utils/
│   │   ├── losses.py             # Contrastive loss functions
│   │   ├── viz.py                # Visualization utilities
│   │   └── utils.py              # General utility functions
│   └── train.py                  # Main training script
├── scripts/
│   └── download_giantsteps.py    # Dataset download script
├── data/
│   └── giantsteps/               # Downloaded dataset (created after setup)
├── requirements.txt              # Python dependencies
├── setup.sh                     # Automated setup script
└── README.md                    # This file
```


## 🎓 Learning Objectives

After completing this tutorial, you will understand:



## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests
- Share your results and experiments

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{music-ssl-ismir2025,
    title={Self-Supervised Learning for Music - an Overview and new horizons},
    author={Julien Guinot, Alain Riou, Marco Pasini, Yuexuan Kong, Gabriel Meseguer-Brocal, Stefan Lattner},
    year={2025},
    howpublished={Tutorial at ISMIR 2025},
    url={https://github.com/your-username/Music-SSL-ISMIR}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The ISMIR community for fostering music information retrieval research
- The authors of the original contrastive learning papers
- The torchaudio and librosa teams for excellent audio processing libraries
- The GiantSteps dataset creators for providing high-quality music data

---

**Happy Learning! 🎵🤖**

For questions or support, please open an issue or contact the tutorial organizers.
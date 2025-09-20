# Self-Supervised Learning for Music Information Retrieval

A comprehensive tutorial on self-supervised learning methods applied to music information retrieval, covering Joint Embedding Architectures, Masked Modeling, Equivariant SSL, and Generative SSL approaches.

## 📖 Book Structure

The tutorial is organized into six main parts:

1. **Part I — Foundations**: Core concepts and background
2. **Part II — Joint Embedding Architectures (JEA)**: Learning similarity representations
3. **Part III — Masked Modeling & JEPA**: Reconstruction and prediction approaches
4. **Part IV — Equivariant SSL**: Handling musical transformations
5. **Part V — Generative SSL**: Combining representation learning with generation
6. **Part VI — Wrapping Up**: Evaluation and future directions

Plus comprehensive appendices covering practical setup and further reading.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Basic familiarity with deep learning and music information retrieval

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ssl-music-tutorial.git
cd ssl-music-tutorial
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build and view the book locally:
```bash
jupyter book build book/
jupyter book serve book/_build/html
```

## 📚 Reading the Tutorial

### Online Version
Visit the online version at: [https://yourusername.github.io/ssl-music-tutorial](https://yourusername.github.io/ssl-music-tutorial)

### Local Development
For local development and contributing:

```bash
# Install development dependencies
pip install -r requirements.txt

# Build the book
jupyter book build book/

# Serve locally for preview
jupyter book serve book/_build/html
```

## 🔧 Hands-On Labs

The tutorial includes several hands-on Jupyter notebooks:

- **Chapter 6**: Training contrastive models (BYOL, SimCLR, VICReg)
- **Chapter 9**: Masked modeling with MAE
- **Chapter 14**: Equivariant SSL for tonality estimation
- **Chapter 17**: Generative SSL applications

## 📊 Datasets

The tutorial uses several music datasets. See [Appendix A: Practical Setup](book/appendices/practical_setup.md) for details on dataset preparation.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Reporting issues
- Suggesting improvements
- Contributing code or content
- Improving documentation

## 📄 License

This tutorial is licensed under the MIT License. See the LICENSE file for details.

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

This tutorial builds upon the excellent work of the self-supervised learning and music information retrieval communities. Special thanks to:

- The authors of key SSL papers (SimCLR, BYOL, VICReg, MAE, etc.)
- The MIR community for datasets and evaluation frameworks
- The Jupyter Book team for the excellent documentation platform

---

*Happy learning! 🎵🤖*


# SpheroSeg: Deep Learning Models for Tumor Spheroid Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository contains the implementation of state-of-the-art deep learning models for tumor spheroid segmentation, accompanying our paper "SpheroSeg: AI-Powered Segmentation Tool for Tumor Spheroids" (under review).

We present multiple architectures (ResUNet variants, HRNet, PSPNet) trained on one of the largest annotated spheroid datasets (32,367 images when combined) with a novel two-stage training strategy achieving IoU scores exceeding 0.95.

## 🏆 Key Features

- **Multiple SOTA Architectures**: MA-ResUNet, CBAM-ResUNet, HRNet, PSPNet
- **Two-Stage Training Strategy**: Pre-training on mixed-quality data, fine-tuning on high-quality annotations
- **Comprehensive Evaluation**: Statistical significance testing with proper non-parametric methods
- **Fast Inference**: HRNet achieves 27ms per image
- **Robust Generalization**: Strong performance on external datasets

## 📊 Performance

| Model | IoU (Internal) | IoU (External) | Inference Time |
|-------|---------------|----------------|----------------|
| MA-ResUNet | **0.950** [0.945, 0.955] | 0.880 [0.868, 0.891] | 229ms |
| ResUNet-Small | 0.929 [0.922, 0.935] | **0.903** [0.890, 0.915] | 58ms |
| HRNet | 0.937 [0.931, 0.943] | 0.888 [0.874, 0.901] | **27ms** |
| PSPNet | 0.932 [0.927, 0.937] | 0.874 [0.858, 0.888] | 46ms |

*95% bootstrap confidence intervals shown in brackets*

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spheroseg-models.git
cd spheroseg-models

# Install dependencies
pip install -r requirements_spheroid.txt
```

### Training

```bash
# Basic training
python src/training/CNN_main_spheroid.py \
    --dataset_path /path/to/data \
    --model resunet_advanced \
    --epochs 100 \
    --batch_size 8

# Two-stage training (recommended)
# 1. Pre-train on mixed dataset
python src/training/CNN_main_spheroid.py \
    --dataset_path /path/to/spheromix \
    --model resunet_advanced \
    --epochs 100 \
    --output_dir outputs/pretrained

# 2. Fine-tune on high-quality dataset
python src/training/CNN_main_spheroid.py \
    --dataset_path /path/to/spherohq \
    --model resunet_advanced \
    --epochs 50 \
    --pretrained_path outputs/pretrained/best_model.pth \
    --lr 1e-5
```

### Evaluation

```bash
# Evaluate models
python src/evaluation/evaluate_models_server.py \
    --dataset-path /path/to/test/data

# Run statistical analysis
python src/analysis/enhanced_statistical_analysis.py
```

## 📁 Repository Structure

```
spheroseg-models/
├── src/
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation pipeline
│   ├── analysis/          # Statistical analysis
│   └── utils/             # Utility functions
├── models/                # Model architectures
│   ├── resunet_advanced.py  # MA-ResUNet
│   ├── resunet_small.py     # Lightweight ResUNet
│   ├── hrnet.py             # HRNet implementation
│   └── pspnet_new.py        # PSPNet variant
├── configs/               # Configuration files
├── results/               # Evaluation results
│   ├── evaluation/        # Model predictions
│   └── statistics/        # Statistical analyses
├── paper/                 # Manuscript and figures
├── docs/                  # Documentation
└── outputs/               # Trained models (excluded from git)
```

## 🧠 Model Architectures

### MA-ResUNet (Multi-Attention ResUNet)
- 66M parameters
- SimAM + Triplet Attention + Self-Attention
- Best overall accuracy (0.950 IoU)

### CBAM-ResUNet
- 60M parameters  
- Convolutional Block Attention Module
- Best generalization to external data

### HRNet
- High-Resolution Network
- Fastest inference (27ms)
- Maintains multi-scale features

### PSPNet
- Pyramid Scene Parsing Network
- Pyramid pooling module
- Robust performance across datasets

## 📈 Statistical Analysis

We provide methodologically correct statistical analysis:
- **Wilcoxon signed-rank test** for non-parametric comparisons
- **Rank-biserial correlation** and **Cliff's delta** for effect sizes
- **Bootstrap confidence intervals** (10,000 iterations)
- **Holm-Bonferroni correction** for multiple testing

Run analysis:
```bash
python src/analysis/enhanced_statistical_analysis.py
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{prusek2025spheroseg,
  title={SpheroSeg: AI-Powered Segmentation Tool for Tumor Spheroids},
  author={Průšek, Michal and Novozámský, Adam and others},
  journal={Under Review},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and collaborations, please contact:
- Michal Průšek - [email]
- Adam Novozámský - [email]

## 🙏 Acknowledgments

- Czech Technical University in Prague
- Czech Academy of Sciences
- University of Chemistry and Technology Prague

---

**Note**: Model weights are not included in this repository due to size constraints. Please contact the authors for access to pre-trained models.
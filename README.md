# SpheroSeg: Deep Learning Models for Tumor Spheroid Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

State-of-the-art deep learning models for tumor spheroid segmentation in microscopy images. This repository implements multiple architectures (ResUNet variants, HRNet, PSPNet, UNet) trained on one of the largest annotated spheroid datasets (32,367 images) achieving IoU scores exceeding 0.93.

## 🏆 Key Features

- **8 Different Architectures**: MA-ResUNet, CBAM-ResUNet, LC-ResUNet, HRNet, PSPNet, UNet, LightM-UNet
- **Two-Stage Training Strategy**: Pre-training on mixed dataset (SpheroMix) + fine-tuning on high-quality data (SpheroHQ)
- **Comprehensive Evaluation**: Statistical significance testing, cross-dataset validation
- **Production Ready**: Fast inference (27ms with HRNet), TTA support, robust data pipeline

## 📊 Performance Summary

| Model | Combined IoU | BxPC-3 IoU | DTS IoU | Inference (ms) |
|-------|-------------|------------|---------|----------------|
| **CBAM-ResUNet** | 0.934 | 0.943 | 0.918 | 200 |
| **MA-ResUNet** | 0.925 | 0.935 | 0.906 | 93 |
| **HRNet** | 0.920 | 0.927 | 0.906 | **27** |
| **UNet** | 0.929 | 0.936 | 0.915 | 44 |

## 📁 Repository Structure

```
SpheroSeg/
├── models/                      # Model architectures
│   ├── resunet_ma.py           # MA-ResUNet with multi-attention
│   ├── resunet_cbam.py         # CBAM-ResUNet
│   ├── resunet_lc.py           # LC-ResUNet (lightweight)
│   ├── resunet_ma_mini.py      # Mini MA-ResUNet variant
│   ├── hrnet.py                # High-Resolution Network
│   ├── pspnet.py               # Pyramid Scene Parsing Network
│   ├── unet.py                 # Standard U-Net
│   └── lightm_unet.py          # Lightweight M-UNet
│
├── src/
│   ├── training/
│   │   └── CNN_main_spheroid.py    # Main training script
│   ├── evaluation/
│   │   └── evaluate_models_server.py # Model evaluation pipeline
│   └── inference/
│       ├── inference.py             # Inference script
│       └── README_INFERENCE.md      # Inference documentation
│
├── scripts/
│   └── training/                # Training shell scripts
│       ├── pretrain_*.sh        # Pre-training scripts for each model
│       └── finetune_*.sh        # Fine-tuning scripts for each model
│
├── results/
│   ├── evaluation_results_*/    # Evaluation outputs
│   ├── statistics/              # Statistical analysis results
│   │   ├── comprehensive_analysis.py
│   │   ├── model_summary_statistics.csv
│   │   ├── topsis_rankings.csv
│   │   └── failure_analysis.csv
│   └── app/                    # Application performance metrics
│
├── model_details/               # Training configs and logs
│   ├── {model}_pretrained/      # Pre-trained model details
│   └── {model}_finetuned/       # Fine-tuned model details
│
├── paper/                       # Academic paper materials
│   ├── manuscript.tex           # LaTeX manuscript
│   └── refs.bib                # Bibliography
│
├── docs/
│   └── PARAMETER_COUNTS.md     # Model parameter analysis
│
├── test_models.py              # Model testing utility
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 8GB+ GPU memory (16GB recommended for MA-ResUNet)

### Installation

```bash
# Clone repository
git clone https://github.com/michalprusek/SpheroSeg.git
cd SpheroSeg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Organize your dataset in the following structure:
```
dataset/
├── train/
│   ├── images/
│   │   ├── image_001.png
│   │   └── ...
│   └── masks/
│       ├── image_001.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Note**: Images and masks should have the same filenames. Masks should be binary (0 for background, 255 for spheroid).

## 🎯 Training

### Basic Training

```bash
python src/training/CNN_main_spheroid.py \
    --dataset_path /path/to/your/dataset \
    --model resunet_cbam \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0002 \
    --output_dir outputs/my_training
```

### Two-Stage Training (Recommended)

1. **Pre-training on larger dataset:**
```bash
python src/training/CNN_main_spheroid.py \
    --dataset_path /path/to/large_dataset \
    --model resunet_cbam \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0002 \
    --output_dir outputs/pretrained
```

2. **Fine-tuning on high-quality dataset:**
```bash
python src/training/CNN_main_spheroid.py \
    --dataset_path /path/to/hq_dataset \
    --model resunet_cbam \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --pretrained_path outputs/pretrained/best_model.pth \
    --output_dir outputs/finetuned
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | Model architecture: `resunet_ma`, `resunet_cbam`, `resunet_lc`, `hrnet`, `pspnet`, `unet`, `lightm_unet` |
| `--batch_size` | 8 | Batch size (adjust based on GPU memory) |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 0.0002 | Learning rate |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation for larger effective batch size |
| `--use_instance_norm` | False | Use instance normalization (recommended for medical images) |
| `--use_tta` | False | Enable test-time augmentation during validation |
| `--use_checkpoint` | False | Use gradient checkpointing to save memory |
| `--find_lr` | False | Run learning rate finder before training |

### Memory-Efficient Training

For large models (MA-ResUNet) with limited GPU memory:
```bash
python src/training/CNN_main_spheroid.py \
    --dataset_path /path/to/dataset \
    --model resunet_ma \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --use_checkpoint \
    --mixed_precision
```

## 📈 Evaluation

### Evaluate Single Model

```bash
python src/evaluation/evaluate_models_server.py \
    --dataset-path /path/to/test/dataset \
    --model-path outputs/my_training/best_model.pth \
    --model-type resunet_cbam
```

### Batch Evaluation (All Models in Directory)

```bash
python src/evaluation/evaluate_models_server.py \
    --dataset-path /path/to/test/dataset \
    --models-dir outputs/ \
    --fair-comparison  # Ensures identical evaluation conditions
```

### Evaluation with TTA

```bash
python src/evaluation/evaluate_models_server.py \
    --dataset-path /path/to/test/dataset \
    --model-path outputs/best_model.pth \
    --use-tta  # 8-fold test-time augmentation
```

### Statistical Analysis

```bash
# Run comprehensive statistical analysis
python results/statistics/comprehensive_analysis.py

# Outputs:
# - model_summary_statistics.csv: Per-model performance metrics
# - topsis_rankings.csv: Multi-criteria decision analysis
# - failure_analysis.csv: Cases where IoU < 0.7
# - tests.csv: Statistical significance tests
```

## 🔮 Inference

### Single Image Inference

```bash
python src/inference/inference.py \
    --model-path outputs/best_model.pth \
    --model-type resunet_cbam \
    --image-path /path/to/image.png \
    --output-path /path/to/output.png
```

### Batch Inference

```bash
python src/inference/inference.py \
    --model-path outputs/best_model.pth \
    --model-type hrnet \
    --input-dir /path/to/images/ \
    --output-dir /path/to/predictions/ \
    --use-tta  # Optional: enable TTA for better accuracy
```

### Inference with Uncertainty Estimation

```bash
python src/inference/inference.py \
    --model-path outputs/best_model.pth \
    --model-type resunet_cbam \
    --input-dir /path/to/images/ \
    --output-dir /path/to/predictions/ \
    --use-tta \
    --save-uncertainty  # Saves uncertainty maps
```

## 🛠️ Model Selection Guide

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| **Maximum Accuracy** | CBAM-ResUNet | Highest IoU (0.934), best on external data |
| **Real-time Processing** | HRNet | 27ms inference, good accuracy (0.920 IoU) |
| **Balanced Performance** | UNet | Good accuracy (0.929), fast (44ms) |
| **Limited GPU Memory** | LC-ResUNet | Lightweight, decent performance |
| **Research/Experimentation** | MA-ResUNet | Most advanced architecture |

## 📝 Advanced Configuration

### Custom Loss Functions

The training script supports combined loss functions:
```python
# In CNN_main_spheroid.py
loss = focal_loss + dice_loss + iou_loss + boundary_loss
```

### Data Augmentation

Strong augmentation pipeline using Albumentations:
- Random rotation (±45°)
- Elastic transform
- Grid distortion
- Gaussian noise
- Brightness/contrast adjustment
- Random scale (0.8-1.2)

### Learning Rate Scheduling

- ReduceLROnPlateau with patience=10
- CosineAnnealingLR option available
- Learning rate finder for optimal LR selection

## 🐛 Troubleshooting

### Out of Memory Errors

1. Reduce batch size
2. Enable gradient checkpointing (`--use_checkpoint`)
3. Use gradient accumulation
4. Use mixed precision training (`--mixed_precision`)

### Poor Convergence

1. Run learning rate finder (`--find_lr`)
2. Check data normalization
3. Verify mask values (should be 0 and 255)
4. Increase epochs or adjust patience

### Slow Training

1. Enable mixed precision (`--mixed_precision`)
2. Use data caching (`--use_cache`)
3. Reduce validation frequency
4. Check I/O bottlenecks

## 📊 Results Visualization

```python
# Visualize predictions
python scripts/visualize_results.py \
    --predictions-dir outputs/predictions/ \
    --ground-truth-dir /path/to/masks/ \
    --output-dir visualizations/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{spheroseg2025,
  title={SpheroSeg: Deep Learning Models for Tumor Spheroid Segmentation},
  author={Průšek, Michal and others},
  journal={Medical Image Analysis},
  year={2025}
}
```

## 📧 Contact

For questions and collaborations:
- **Michal Průšek** (Main Author) - prusemic@cvut.cz
- **Adam Novozámský** (Computer Science Lead) - novozamsky@utia.cas.cz
- For bug reports or feature requests, please create a [GitHub Issue](https://github.com/michalprusek/SpheroSeg/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Czech Academy of Sciences, Institute of Information Theory and Automation
- University of Chemistry and Technology Prague
- All contributors to the SpheroSeg dataset

---

## 📦 Pre-trained Model Weights

Pre-trained model weights for all architectures are available for download:

🔗 **Download link**: [https://staff.utia.cas.cz/novozada/spheroseg/](https://staff.utia.cas.cz/novozada/spheroseg/)

The weights include both pretrained and fine-tuned versions for each model:
- `resunet_ma_pretrained.pth` / `resunet_ma_finetuned.pth` - MA-ResUNet
- `resunet_cbam_pretrained.pth` / `resunet_cbam_finetuned.pth` - CBAM-ResUNet
- `resunet_lc_pretrained.pth` / `resunet_lc_finetuned.pth` - LC-ResUNet
- `hrnet_pretrained.pth` / `hrnet_finetuned.pth` - HRNet
- `pspnet_pretrained.pth` / `pspnet_finetuned.pth` - PSPNet
- `unet_pretrained.pth` / `unet_finetuned.pth` - UNet
- `lightm_unet_pretrained.pth` / `lightm_unet_finetuned.pth` - LightM-UNet

Place downloaded weights in the `outputs/` directory for evaluation or inference.
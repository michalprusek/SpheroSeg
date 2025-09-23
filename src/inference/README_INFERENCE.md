# Spheroid Segmentation Inference

## Overview
The `inference.py` script provides a unified interface for running inference with all trained spheroid segmentation models. It supports single image or batch processing, Test Time Augmentation (TTA), and various output formats.

## Installation
```bash
# Ensure you have the required dependencies
pip install torch torchvision opencv-python pillow tqdm numpy
```

## Basic Usage

### Single Image Inference
```bash
python inference.py \
    --model hrnet \
    --weights /Volumes/T7/SpheroSeg_upload/weights/hrnet_pretrained.pth \
    --input /path/to/image.png \
    --output /path/to/output_folder
```

### Batch Processing (Folder of Images)
```bash
python inference.py \
    --model resunet_cbam \
    --weights /Volumes/T7/SpheroSeg_upload/weights/resunet_cbam_pretrained.pth \
    --input /path/to/image_folder \
    --output /path/to/output_folder
```

### With Test Time Augmentation (8-fold)
```bash
python inference.py \
    --model resunet_ma \
    --weights /path/to/weights.pth \
    --input /path/to/images \
    --output /path/to/output \
    --use-tta
```

### Save Overlay Visualizations
```bash
python inference.py \
    --model unet \
    --weights /path/to/weights.pth \
    --input /path/to/images \
    --output /path/to/output \
    --save-overlay
```

## Available Models

| Model Name | Architecture | Description | Speed |
|------------|--------------|-------------|-------|
| `hrnet` | HRNetV2 | Fastest model, good accuracy | ~27ms |
| `unet` | Standard U-Net | Classic architecture | ~44ms |
| `pspnet` | PSPNet | Pyramid pooling | ~46ms |
| `resunet_lc` | Lightweight CBAM ResUNet | Best on external data | ~55ms |
| `resunet_cbam` | ResUNet with CBAM | Strong performer | ~93ms |
| `resunet_ma` | Multi-Attention ResUNet | High accuracy | ~93ms |
| `lightm_unet` | Lightweight Mamba U-Net | Experimental | ~200ms |

## Default Paths

- **Weights**: `/Volumes/T7/SpheroSeg_upload/weights/`
- **Models**: `/Users/michalprusek/Desktop/spheroseg_models/models/`
- **Test Images**: `/Users/michalprusek/Desktop/spheroseg_models/DATASETS/DTS/images/`

## Output Format

The script generates:
1. **Binary masks** (`*_mask.png`): 8-bit grayscale images (0=background, 255=spheroid)
2. **Overlay images** (`*_overlay.png`): Original image with green mask overlay (optional)
3. **Results JSON** (`inference_results.json`): Processing statistics and metadata

### Results JSON Structure
```json
{
  "total_images": 10,
  "total_time": 2.34,
  "mean_time": 0.234,
  "std_time": 0.012,
  "median_time": 0.230,
  "results": [
    {
      "image": "/path/to/image.png",
      "mask": "/path/to/output/image_mask.png",
      "processing_time": 0.234,
      "mask_area": 0.156
    }
  ]
}
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | Model architecture name |
| `--weights` | str | `/Volumes/T7/SpheroSeg_upload/weights` | Path to weights file or directory |
| `--input` | str | `/Users/michalprusek/Desktop/spheroseg_models/DATASETS/DTS/images` | Input image or folder |
| `--output` | str | required | Output folder for masks |
| `--use-tta` | flag | False | Enable Test Time Augmentation |
| `--save-overlay` | flag | False | Save overlay visualizations |
| `--device` | str | cuda | Device (cuda/cpu) |
| `--img-size` | int | 1024 | Model input size |

## Examples

### 1. Quick Test with HRNet (Fastest)
```bash
python inference.py \
    --model hrnet \
    --weights /Volumes/T7/SpheroSeg_upload/weights/hrnet_pretrained.pth \
    --input ./test_image.png \
    --output ./output_hrnet
```

### 2. High Accuracy with CBAM-ResUNet + TTA
```bash
python inference.py \
    --model resunet_cbam \
    --weights /Volumes/T7/SpheroSeg_upload/weights/resunet_cbam_pretrained.pth \
    --input ./test_images/ \
    --output ./output_cbam \
    --use-tta \
    --save-overlay
```

### 3. CPU-only Processing
```bash
python inference.py \
    --model unet \
    --weights ./weights/unet.pth \
    --input ./images/ \
    --output ./masks/ \
    --device cpu
```

### 4. Auto-detect Weights from Directory
```bash
# If weights directory contains model-specific files like "hrnet_pretrained.pth"
python inference.py \
    --model hrnet \
    --weights /Volumes/T7/SpheroSeg_upload/weights/ \
    --input ./images/ \
    --output ./masks/
```

## Performance Tips

1. **For speed**: Use HRNet without TTA
2. **For accuracy**: Use ResUNet-CBAM or ResUNet-MA with TTA
3. **For generalization**: Use ResUNet-LC (best on external data)
4. **Batch processing**: The script processes images sequentially to manage GPU memory
5. **Large datasets**: Consider splitting into batches and running in parallel

## Troubleshooting

### CUDA Out of Memory
- Reduce `--img-size` (e.g., 512 instead of 1024)
- Use CPU mode with `--device cpu`
- Process smaller batches

### Model Loading Errors
- Ensure weights match the model architecture
- Check that the weights file exists and is accessible
- Verify model name is correct (see Available Models table)

### Missing Dependencies
```bash
pip install torch torchvision opencv-python-headless pillow tqdm numpy
```

## Integration Example

```python
from inference import ModelInference, process_images
from pathlib import Path

# Initialize model
model = ModelInference(
    model_name='hrnet',
    weights_path='/path/to/weights.pth',
    device='cuda'
)

# Process images
stats = process_images(
    model_inference=model,
    input_path=Path('./images'),
    output_path=Path('./output'),
    use_tta=False,
    save_overlay=True
)

print(f"Processed {stats['total_images']} images")
print(f"Mean time: {stats['mean_time']*1000:.2f} ms")
```
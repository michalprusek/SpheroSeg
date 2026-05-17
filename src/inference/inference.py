#!/usr/bin/env python3
"""
Inference script for spheroid segmentation models.
Supports single image or batch processing from folders.

Usage:
    python inference.py \
        --model resunet_cbam \
        --weights /path/to/best_model.pth \
        --input /path/to/image_or_folder \
        --output /path/to/output_folder

Inputs may be a single image file or a directory; output is always a directory.
For batch evaluation against ground-truth masks, use
`scripts/a3/evaluate_a3.py` instead — this script is for production inference
only and does not compute IoU.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
from tqdm import tqdm
import time
from datetime import datetime

# Make `models/` package importable when running this file directly from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import all model architectures
from models.resunet_ma import AdvancedResUNet
from models.resunet_lc import ResUNetSmall
from models.resunet_cbam import ResUNetCBAM
from models.unet import UNet
from models.hrnet import HRNetV2
from models.pspnet import PSPNet
from models.lightm_unet import LightMUNet


# Model factory mapping
MODEL_REGISTRY = {
    'resunet_ma': AdvancedResUNet,
    'resunet_advanced': AdvancedResUNet,  # alias
    'resunet_lc': ResUNetSmall,
    'resunet_small': ResUNetSmall,  # alias
    'resunet_cbam': ResUNetCBAM,
    'unet': UNet,
    'hrnet': HRNetV2,
    'pspnet': PSPNet,
    'lightm_unet': LightMUNet,
    'lightmunet': LightMUNet,  # alias
}


class InferenceDataset(Dataset):
    """Dataset for inference on images."""
    
    def __init__(self, image_paths: List[Path], img_size: int = 1024):
        self.image_paths = image_paths
        self.img_size = img_size
        
        # Standard ImageNet normalization (same as training)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        # Resize to model input size
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        
        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)
        
        return {
            'image': image_tensor,
            'path': str(img_path),
            'original_size': original_size
        }


class ModelInference:
    """Handles model loading and inference."""
    
    def __init__(self, model_name: str, weights_path: str, device: str = 'cuda'):
        self.model_name = model_name.lower()
        self.weights_path = Path(weights_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded {self.model_name} model from {self.weights_path}")
        print(f"✓ Using device: {self.device}")
        
    def _load_model(self) -> nn.Module:
        """Load model architecture and weights."""
        
        # Get model class
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        
        model_class = MODEL_REGISTRY[self.model_name]
        
        # Load checkpoint (weights_only=False for compatibility with saved configs)
        checkpoint = torch.load(self.weights_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                config = checkpoint.get('config', {})
            else:
                state_dict = checkpoint
                config = {}
        else:
            state_dict = checkpoint
            config = {}
        
        # Initialize model with appropriate config
        if self.model_name == 'hrnet':
            # HRNet expects n_class parameter (from line 236: def __init__(self, n_class=1, ...))
            model = model_class(n_class=1, pretrained=False, use_instance_norm=True)
        elif self.model_name == 'pspnet':
            # PSPNet initialization - complex due to multiple variants
            # Based on server script logic, we need to check config in checkpoint
            use_instance_norm = True  # Default from training configs  
            backbone = 'resnet101'  # Default backbone
            
            # Try to get config from checkpoint itself
            if 'use_instance_norm' in config:
                use_instance_norm = config['use_instance_norm']
            if 'backbone' in config:
                backbone = config['backbone']
                
            # PSPNet trained models typically use instance norm
            model = model_class(n_class=1, backbone=backbone, pretrained=False, use_instance_norm=use_instance_norm)
        elif self.model_name == 'resunet_ma' and 'mini' in str(self.weights_path):
            # Mini version with reduced channels
            model = model_class(
                in_channels=3, 
                out_channels=1,
                features=[20, 40, 80, 160]  # Mini architecture
            )
        elif self.model_name == 'lightm_unet':
            # LightM-UNet doesn't have parameters in __init__ (uses default)
            model = model_class()
        else:
            # Standard initialization for ResUNet variants and UNet
            try:
                model = model_class(in_channels=3, out_channels=1)
            except TypeError:
                # Fallback if model doesn't accept parameters
                model = model_class()
        
        # Clean state dict keys (remove module. prefix if present)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[7:]] = v
            else:
                cleaned_state_dict[k] = v
        
        # Load weights with strict=False to handle minor architecture differences
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        return model
    
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Run inference on a single image tensor."""
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Handle models that return tuple (like PSPNet)
        if isinstance(output, tuple):
            output = output[0]
        
        # Apply sigmoid and threshold
        output = torch.sigmoid(output)
        mask = (output > 0.5).float()
        
        # Convert to numpy and remove batch dimension
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        
        return mask
    
    @torch.no_grad()
    def predict_with_tta(self, image_tensor: torch.Tensor, augmentations: int = 8) -> np.ndarray:
        """Run inference with Test Time Augmentation."""
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        predictions = []
        
        # Original
        predictions.append(self.predict(image_tensor))
        
        if augmentations >= 4:
            # Rotations
            for angle in [90, 180, 270]:
                rotated = torch.rot90(image_tensor, k=angle//90, dims=[2, 3])
                pred = self.predict(rotated)
                # Rotate back
                pred = np.rot90(pred, k=-angle//90)
                predictions.append(pred)
        
        if augmentations >= 8:
            # Flips
            for flip_dims in [[2], [3], [2, 3]]:
                flipped = torch.flip(image_tensor, dims=flip_dims)
                pred = self.predict(flipped)
                # Flip back
                if 2 in flip_dims:
                    pred = np.flip(pred, axis=0)
                if 3 in flip_dims:
                    pred = np.flip(pred, axis=1)
                predictions.append(pred)
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        return (avg_prediction > 0.5).astype(np.float32)


def process_images(
    model_inference: ModelInference,
    input_path: Path,
    output_path: Path,
    use_tta: bool = False,
    save_overlay: bool = False
) -> Dict:
    """Process single image or folder of images."""
    
    # Collect image paths
    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Get all image files from folder
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_paths = [
            p for p in input_path.glob('*')
            if p.suffix.lower() in image_extensions
        ]
        image_paths.sort()
    
    if not image_paths:
        raise ValueError(f"No images found in {input_path}")
    
    print(f"Found {len(image_paths)} images to process")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = InferenceDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Process images
    results = []
    processing_times = []
    
    for batch in tqdm(dataloader, desc="Processing images"):
        start_time = time.perf_counter()
        
        # Get image data
        image_tensor = batch['image']
        img_path = Path(batch['path'][0])
        original_size = batch['original_size']
        
        # Run inference
        if use_tta:
            mask = model_inference.predict_with_tta(image_tensor)
        else:
            mask = model_inference.predict(image_tensor)
        
        # Resize mask to original size
        mask_resized = cv2.resize(mask, (original_size[0].item(), original_size[1].item()), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Save mask
        mask_filename = img_path.stem + '_mask.png'
        mask_path = output_path / mask_filename
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_uint8)
        
        # Save overlay if requested
        if save_overlay:
            original_img = cv2.imread(str(img_path))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = original_img.copy()
            mask_colored = np.zeros_like(original_img)
            mask_colored[:, :, 1] = mask_uint8  # Green channel for mask
            overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)
            
            overlay_filename = img_path.stem + '_overlay.png'
            overlay_path = output_path / overlay_filename
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Record processing time
        processing_time = time.perf_counter() - start_time
        processing_times.append(processing_time)
        
        # Store result
        results.append({
            'image': str(img_path),
            'mask': str(mask_path),
            'processing_time': processing_time,
            'mask_area': np.sum(mask_resized) / (mask_resized.shape[0] * mask_resized.shape[1])
        })
    
    # Calculate statistics
    stats = {
        'total_images': len(results),
        'total_time': sum(processing_times),
        'mean_time': np.mean(processing_times),
        'std_time': np.std(processing_times),
        'median_time': np.median(processing_times),
        'results': results
    }
    
    # Save results JSON
    results_json_path = output_path / 'inference_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description='Spheroid Segmentation Inference')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture to use')
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth) or directory containing one')

    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or folder of images')
    
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder for masks')
    
    # Processing arguments
    parser.add_argument('--use-tta', action='store_true',
                        help='Use Test Time Augmentation (8-fold)')
    
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save overlay images showing mask on original')
    
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    
    parser.add_argument('--img-size', type=int, default=1024,
                        help='Image size for model input')
    
    args = parser.parse_args()
    
    # Process paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    weights_path = Path(args.weights)
    
    # Handle weights path - if directory, look for model-specific weights
    if weights_path.is_dir():
        # Look for model-specific weights file
        possible_names = [
            f"{args.model}.pth",
            f"{args.model}_best.pth",
            f"{args.model}_pretrained.pth",
            f"{args.model}_finetuned.pth",
            "best_model.pth",
            "model.pth"
        ]
        
        weights_file = None
        for name in possible_names:
            candidate = weights_path / name
            if candidate.exists():
                weights_file = candidate
                break
        
        if weights_file is None:
            # List available weights
            available = list(weights_path.glob('*.pth'))
            if available:
                print(f"Available weights in {weights_path}:")
                for w in available:
                    print(f"  - {w.name}")
                raise ValueError(f"No suitable weights found for model '{args.model}'")
            else:
                raise ValueError(f"No .pth files found in {weights_path}")
        
        weights_path = weights_file
    
    # Validate input path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    print(f"\n{'='*60}")
    print(f"Spheroid Segmentation Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Weights: {weights_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"TTA: {'Enabled' if args.use_tta else 'Disabled'}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Initialize model
    model_inference = ModelInference(args.model, str(weights_path), device=args.device)
    
    # Process images
    stats = process_images(
        model_inference,
        input_path,
        output_path,
        use_tta=args.use_tta,
        save_overlay=args.save_overlay
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"{'='*60}")
    print(f"Processed: {stats['total_images']} images")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Mean time per image: {stats['mean_time']*1000:.2f} ms")
    print(f"Output saved to: {output_path}")
    print(f"Results JSON: {output_path / 'inference_results.json'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
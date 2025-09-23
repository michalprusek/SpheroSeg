#!/usr/bin/env python3
"""
Evaluační skript optimalizovaný pro server s GPU
Testuje natrénované modely na test setu: /data/prusek/

AUTOMATIC MODEL DISCOVERY:
- Automatically finds the latest/best trained models for each architecture
- Searches in multiple output directories for the most recent training runs
- Supports: pspnet_new, resunet, resunet_small, hrnet, resunet_advanced, unet, cbam_unet
- Loads best_model.pth from the most recent training run for each model type
- Handles missing models gracefully with proper error reporting

FEATURES:
- Automatic model discovery and loading
- Robust error handling for architecture mismatches
- Configuration loading from config.json files
- Comprehensive evaluation metrics (IoU, Dice, F1, etc.)
- Optimalized performance timing measurements
- Detailed results export (JSON + CSV)

OPTIMALIZACE MĚŘENÍ RYCHLOSTI:
- Warm-up 30 iterací pro stabilizaci GPU před měřením
- Separátní měření rychlosti na 100 iteracích pro přesné statistiky
- torch.cuda.synchronize() před/po každém měření pro eliminaci async operací
- Fixní batch_size=1 pro konzistentní měření
- num_workers=0 pro eliminaci I/O variability
- pin_memory=True pro rychlejší GPU transfer
- Inference mód bez augmentací pro reprodukovatelné výsledky
- time.perf_counter() pro přesnější časování
- Detailní statistiky: mean, std, median, min, max, P95, P99, CV%
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from datetime import datetime

# Import model architektur
sys.path.append('/home/prusek/SpheroSeg/NN/diplomka')
from models.resunet import ResUNet
from models.resunet_small import ResUNetSmall
from models.hrnet import HRNetV2
from models.pspnet_stable import PSPNet
from models.transunet import TransUNet
from models.resunet_advanced import AdvancedResUNet
from models.unet import UNet
from models.resunet_cbam import ResUNetCBAM

class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset pro načítání obrázků a masek optimalizovaný pro inference"""

    def __init__(self, images_dir, masks_dir, transform=None, inference_mode=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.inference_mode = inference_mode  # Vypne augmentace pro stabilní měření

        # Najdi všechny obrázky (vyfiltruj skryté soubory)
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        self.image_files = []

        for ext in image_extensions:
            files = list(self.images_dir.glob(f'*{ext}')) + list(self.images_dir.glob(f'*{ext.upper()}'))
            # Filtruj skryté soubory (začínající tečkou nebo ._)
            files = [f for f in files if not f.name.startswith('.') and not f.name.startswith('._')]
            self.image_files.extend(files)

        # Najdi odpovídající masky
        self.valid_pairs = []
        for img_file in self.image_files:
            mask_file = self.find_mask_file(img_file)
            if mask_file and mask_file.exists():
                self.valid_pairs.append((img_file, mask_file))

        print(f"Nalezeno {len(self.valid_pairs)} platných párů obrázek-maska")
        if self.inference_mode:
            print("Dataset v inference módu - bez augmentací pro stabilní měření")
    
    def find_mask_file(self, img_file):
        """Najdi odpovídající soubor masky"""
        base_name = img_file.stem
        mask_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        
        for ext in mask_extensions:
            mask_file = self.masks_dir / f"{base_name}{ext}"
            if mask_file.exists():
                return mask_file
            mask_file = self.masks_dir / f"{base_name}{ext.upper()}"
            if mask_file.exists():
                return mask_file
        return None
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]

        # Načti obrázek
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Načti masku
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # Normalizuj masku na 0-1
        mask = (mask > 128).astype(np.float32)

        # Resize na 1024x1024 - použij INTER_LINEAR pro konzistentní výsledky
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Převeď na tensor a aplikuj ImageNet normalizaci (stejně jako při tréninku)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # ImageNet normalizace - KRITICKÉ pro správnou funkci modelů!
        # Toto musí být stejné jako při tréninku v CNN_main_spheroid.py
        if self.inference_mode:
            # V inference módu použij přesnou normalizaci bez augmentací
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
        else:
            # Pokud není inference mód, aplikuj transform (pro případné augmentace)
            if self.transform:
                image = self.transform(image)
            else:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                image = normalize(image)

        mask = torch.from_numpy(mask).float()

        return image, mask, str(img_path.name)

def find_latest_model_checkpoint(models_base_paths, model_type, prefer_pretrained=False):
    """
    Find the latest/best checkpoint for a given model type.

    Args:
        models_base_paths: List of base paths to search for models
        model_type: Type of model to find ('pspnet_new', 'resunet', 'resunet_small', 'hrnet', 'resunet_advanced')
        prefer_pretrained: If True, prefer pretrained models over finetuned ones

    Returns:
        tuple: (model_path, model_info) or (None, None) if not found
    """
    print(f"  Hledám nejnovější checkpoint pro model: {model_type} ({'pretrained' if prefer_pretrained else 'finetuned'})")

    # Define search patterns for each model type
    # Order matters - first patterns have higher priority
    if prefer_pretrained:
        search_patterns = {
            'pspnet_new': [
                r'pspnet_new.*pretrained.*',
                r'pspnet_new_pretrained.*'
            ],
            'resunet': [
                r'resunet_pretrained$',
                r'resunet(?!_small)(?!_advanced).*pretrained.*'  # resunet but not resunet_small or resunet_advanced
            ],
            'resunet_small': [
                r'resunet_small.*pretrain.*',  # matches "pretrain" in directory names
                r'resunet_small.*pretrained.*'
            ],
            'hrnet': [
                r'hrnet_pretrained$',
                r'hrnet.*pretrained.*'
            ],
            'resunet_advanced': [
                r'resunet_advanced_pretrained$',
                r'resunet_advanced.*pretrained.*'
            ],
            'unet': [
                r'unet_pretrain.*',
                r'unet.*pretrain.*'
            ],
            'cbam_unet': [
                r'resunet_cbam_pretrain.*',
                r'resunet_cbam.*pretrain.*'
            ]
        }
    else:
        search_patterns = {
            'pspnet_new': [
                r'pspnet_new.*finetune.*',
                r'pspnet_new.*finetuned.*'
            ],
            'resunet': [
                r'resunet_finetuned$',
                r'resunet_spheroid$',
                r'resunet(?!_small)(?!_advanced).*finetuned.*',  # resunet but not resunet_small or resunet_advanced
                r'resunet(?!_small)(?!_advanced)(?!_).*'  # resunet but not resunet_small or resunet_advanced
            ],
            'resunet_small': [
                r'resunet_small.*finetune.*',  # matches "finetune" in directory names
                r'resunet_small.*finetuned.*'
            ],
            'hrnet': [
                r'hrnet_finetuned$',
                r'hrnet.*finetuned.*'
            ],
            'resunet_advanced': [
                r'resunet_advanced_finetuned$',
                r'resunet_advanced.*finetuned.*'
            ],
            'unet': [
                r'unet_finetune.*',
                r'unet.*finetune.*'
            ],
            'cbam_unet': [
                r'resunet_cbam_finetune.*',
                r'resunet_cbam.*finetune.*'
            ]
        }

    if model_type not in search_patterns:
        print(f"    Neznámý typ modelu: {model_type}")
        return None, None

    best_checkpoint = None
    best_info = None
    latest_time = None

    # Search in all provided base paths
    for base_path in models_base_paths:
        if not base_path.exists():
            print(f"    Cesta neexistuje: {base_path}")
            continue

        print(f"    Prohledávám: {base_path}")

        # Get all directories in the base path
        for dir_path in base_path.iterdir():
            if not dir_path.is_dir():
                continue

            dir_name = dir_path.name

            # Check if directory matches any pattern for this model type
            matches = False
            for pattern in search_patterns[model_type]:
                if re.match(pattern, dir_name, re.IGNORECASE):
                    matches = True
                    break

            if not matches:
                continue

            print(f"      Nalezena složka: {dir_name}")

            # Look for best_model.pth
            checkpoint_path = dir_path / 'best_model.pth'
            if not checkpoint_path.exists():
                print(f"        Nenalezen best_model.pth")
                continue

            # Get modification time or extract timestamp from directory name
            dir_time = None

            # Try to extract timestamp from directory name (format: YYYYMMDD_HHMMSS)
            timestamp_match = re.search(r'(\d{8}_\d{6})', dir_name)
            if timestamp_match:
                try:
                    dir_time = datetime.strptime(timestamp_match.group(1), '%Y%m%d_%H%M%S')
                except ValueError:
                    pass

            # Fallback to file modification time
            if dir_time is None:
                dir_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)

            print(f"        Čas: {dir_time}")

            # Check if this is the latest
            if latest_time is None or dir_time > latest_time:
                latest_time = dir_time
                best_checkpoint = checkpoint_path
                best_info = {
                    'directory': dir_name,
                    'timestamp': dir_time,
                    'path': str(checkpoint_path)
                }
                print(f"        -> Nový nejnovější checkpoint")

    if best_checkpoint:
        print(f"    Nejnovější checkpoint: {best_info['directory']} ({best_info['timestamp']})")
        return best_checkpoint, best_info
    else:
        print(f"    Nenalezen žádný checkpoint pro {model_type}")
        return None, None



def detect_features_from_checkpoint(checkpoint_path):
    """
    Detect feature dimensions from checkpoint when missing from config.json

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        list: Detected feature dimensions or None if detection fails
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Method 1: Look for downs/encoder layers (most reliable)
        downs_features = []
        for i in range(6):  # Check up to 6 levels (downs.0 to downs.5)
            # Try different naming patterns
            for pattern in [f'downs.{i}.conv1.weight', f'downs.{i}.0.conv1.weight', f'encoder{i+1}.0.conv1.weight']:
                if pattern in state_dict:
                    feature_dim = state_dict[pattern].shape[0]
                    downs_features.append(feature_dim)
                    break

        if downs_features:
            print(f"    Detected from downs/encoder layers: {downs_features}")
            return downs_features

        # Method 2: Look for init_conv and build standard pyramid
        init_conv_out = None
        if 'init_conv.double_conv.0.weight' in state_dict:  # UNet style
            init_conv_out = state_dict['init_conv.double_conv.0.weight'].shape[0]
        elif 'init_conv.0.weight' in state_dict:  # Other models
            init_conv_out = state_dict['init_conv.0.weight'].shape[0]

        if init_conv_out:
            print(f"    Detected init_conv: {init_conv_out}")
            # Build standard pyramid from init_conv
            if init_conv_out == 20:
                return [20, 40, 80, 160, 320]  # Mini configuration (5 levels)
            elif init_conv_out == 32:
                return [32, 64, 128, 256, 512]  # Standard configuration (5 levels)
            elif init_conv_out == 48:
                return [48, 96, 192, 384, 512]  # ResUNetSmall configuration (5 levels)
            elif init_conv_out == 64:
                # For UNet with init_conv=64, always use default UNet features
                # This matches the training script: UNet(dropout_rate=0.1) uses default features=[64, 128, 256, 512, 1024]
                return [64, 128, 256, 512, 1024]  # Standard UNet (5 levels)
            else:
                # Build pyramid with doubling pattern (assume 5 levels)
                features = [init_conv_out]
                current = init_conv_out
                for _ in range(4):
                    current *= 2
                    features.append(current)
                return features



        print(f"    Could not detect features from checkpoint")
        return None

    except Exception as e:
        print(f"    Error detecting features from checkpoint: {e}")
        return None


def load_model_config(model_dir):
    """
    Load model configuration from config.json - REQUIRED for evaluation

    Args:
        model_dir: Directory containing the model files

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found
        ValueError: If config.json cannot be parsed or is missing required fields
    """
    config_path = model_dir / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}. Cannot evaluate model without training configuration.")

    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"    Loaded config from {config_path}")

        # Validate required fields
        required_fields = ['model', 'img_size', 'use_instance_norm']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"config.json missing required fields: {missing_fields}")

        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config.json: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config.json: {e}")

def create_model_from_config(model_name, model_path, device):
    """
    Create model using exact parameters from config.json

    Args:
        model_name: Type of model to create
        model_path: Path to checkpoint file
        device: Device to create model on

    Returns:
        tuple: (model, config_used)

    Raises:
        ValueError: If model cannot be created from config
    """
    model_dir = Path(model_path).parent
    config = load_model_config(model_dir)

    # Extract parameters from config
    use_instance_norm = config['use_instance_norm']
    img_size = config['img_size']

    # For evaluation, always disable dropout (set to 0)
    # Dropout during evaluation causes randomness and is not fair for comparison
    dropout_rate = 0.0  # Always disable dropout for evaluation

    # Extract base model name if it has suffix
    base_model_name = model_name
    if model_name.endswith('_finetuned') or model_name.endswith('_pretrained'):
        base_model_name = model_name.rsplit('_', 1)[0]

    # Map mini models to their base architecture
    if 'resunet_advanced_mini' in base_model_name:
        base_model_name = 'resunet_advanced'

    # Get model-specific parameters from config
    features = config.get('features', None)
    backbone = config.get('backbone', 'resnet101')  # Default for PSPNet

    # If features are missing from config, try to detect them from checkpoint
    if features is None and base_model_name in ['resunet', 'resunet_advanced', 'resunet_small', 'unet', 'cbam_unet']:
        print(f"    Features not found in config, detecting from checkpoint...")
        try:
            features = detect_features_from_checkpoint(model_path)
            if features:
                print(f"    Detected features from checkpoint: {features}")
            else:
                print(f"    Could not detect features from checkpoint, using defaults")
        except Exception as e:
            print(f"    Warning: Failed to detect features from checkpoint: {e}")
            print(f"    Using default features for {base_model_name}")

    print(f"    Creating {base_model_name} model from config parameters...")

    try:
        # Create model based on type using config parameters
        if base_model_name == 'resunet':
            # Use default features if not specified in config
            if features is None:
                features = [24, 48, 96, 192, 256]  # Default ResUNet features
            model = ResUNet(
                in_channels=3,
                out_channels=1,
                features=features,
                use_instance_norm=use_instance_norm,
                dropout_rate=dropout_rate
            )
        elif base_model_name == 'resunet_advanced':
            # Use default features if not specified in config
            if features is None:
                features = [32, 64, 128, 256, 512]  # Default AdvancedResUNet features
            model = AdvancedResUNet(
                in_channels=3,
                out_channels=1,
                features=features,
                use_instance_norm=use_instance_norm,
                dropout_rate=dropout_rate
            )
        elif base_model_name == 'resunet_small':
            # Use default features if not specified in config
            if features is None:
                features = [48, 96, 192, 384, 512]  # Default ResUNetSmall features
            model = ResUNetSmall(
                in_channels=3,
                out_channels=1,
                features=features,
                use_instance_norm=use_instance_norm,
                dropout_rate=dropout_rate
            )
        elif base_model_name == 'unet':
            # Use default features if not specified in config
            if features is None:
                features = [64, 128, 256, 512, 1024]  # Default UNet features
            model = UNet(
                in_channels=3,
                out_channels=1,
                features=features,
                use_instance_norm=use_instance_norm,
                dropout_rate=dropout_rate
            )
        elif base_model_name == 'cbam_unet':
            # Use default features if not specified in config
            if features is None:
                features = [64, 128, 256, 512]  # Default CBAM UNet features (4 levels)
            model = ResUNetCBAM(
                in_channels=3,
                out_channels=1,
                features=features,
                use_instance_norm=use_instance_norm,
                dropout_rate=dropout_rate
            )
        elif base_model_name == 'hrnet':
            model = HRNetV2(n_class=1, use_instance_norm=use_instance_norm)
        elif base_model_name == 'pspnet':
            model = PSPNet(n_class=1, use_instance_norm=use_instance_norm)
        elif base_model_name == 'pspnet_new':
            try:
                from models.pspnet_new import PSPNet as NewPSPNet
                model = NewPSPNet(n_class=1, backbone=backbone, pretrained=False, use_instance_norm=use_instance_norm)
            except ImportError:
                print("    Warning: pspnet_new not found, using pspnet_stable")
                from models.pspnet_stable import PSPNet
                model = PSPNet(n_class=1, use_instance_norm=use_instance_norm)
        elif base_model_name == 'transunet':
            model = TransUNet(in_channels=3, out_channels=1, img_size=img_size, use_instance_norm=use_instance_norm)
        elif 'lightm_unet' in model_name:
            from models.lightm_unet import LightMUNet
            base_channels = config.get('base_channels', 32)
            encoder_layers = config.get('encoder_layers', [1, 2, 2])
            decoder_layers = config.get('decoder_layers', [2, 2, 1])
            bottleneck_layers = config.get('bottleneck_layers', 4)
            model = LightMUNet(
                in_channels=3,
                out_channels=1,
                base_channels=base_channels,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                bottleneck_layers=bottleneck_layers,
                dropout_rate=dropout_rate,
                use_instance_norm=use_instance_norm
            )
        else:
            raise ValueError(f"Unknown model type: {base_model_name}")

        print(f"    ✅ Successfully created {base_model_name} model from config")
        return model, {'name': 'config_based', 'features': features}

    except Exception as e:
        raise ValueError(f"Failed to create {base_model_name} model from config: {e}")





def load_model(model_name, model_path, device):
    """Load model using exact configuration from config.json"""
    print(f"  Loading model {model_name}...")

    try:
        # Create model from config
        model, config_used = create_model_from_config(model_name, model_path, device)
        print(f"    Successfully created model with {config_used['name']} configuration")
    except Exception as e:
        raise ValueError(f"Failed to create model {model_name} from config: {e}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # Load checkpoint - Safe checkpoint loading for PyTorch 2.6+
    try:
        # Try with weights_only=True first (safer) with safe globals for argparse.Namespace
        import argparse
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"    Warning: weights_only=True with safe globals failed ({e}), trying weights_only=False...")
        # Fallback to weights_only=False (trusted source)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract state dict and metadata
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'epoch' in checkpoint:
            print(f"    Model trained for {checkpoint['epoch']} epochs")
        if 'best_val_loss' in checkpoint:
            print(f"    Best validation loss: {checkpoint['best_val_loss']:.4f}")
        if 'best_iou' in checkpoint:
            print(f"    Best IoU: {checkpoint['best_iou']:.4f}")
    else:
        state_dict = checkpoint

    # Load state dict with strict=True to ensure exact match
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"    Model weights loaded successfully (exact match)")
    except Exception as e:
        print(f"    ❌ Error loading model weights with strict=True: {e}")
        print(f"    This indicates architecture mismatch between config and checkpoint.")
        print(f"    Please verify that the config.json matches the training configuration.")
        raise ValueError(f"Failed to load weights for {model_name}: {e}")

    model.to(device)
    model.eval()
    
    # KRITICKÉ: Explicitně vypnout dropout pro spravedlivou evaluaci
    # Dropout při evaluaci způsobuje náhodnost a horší výsledky
    print("    Disabling dropout for evaluation...")
    dropout_disabled_count = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.p = 0.0  # Vypnout dropout úplně
            module.training = False  # Zajistit eval mód
            dropout_disabled_count += 1
    
    if dropout_disabled_count > 0:
        print(f"    Disabled {dropout_disabled_count} dropout layers")
    
    # Verify model is in eval mode and check normalization layers
    for module in model.modules():
        if hasattr(module, 'training'):
            assert not module.training, f"Module {module} is not in eval mode!"
        # Zkontrolovat že normalizační vrstvy jsou také v eval módu
        if isinstance(module, (torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d)):
            assert not module.training, f"Normalization {module} is not in eval mode!"

    print(f"    Model {model_name} loaded successfully")
    return model

def validate_model_loading(model, device, model_name):
    """
    Validate that the model can perform forward pass correctly

    Args:
        model: The loaded model
        device: Device to run validation on
        model_name: Name of the model for logging

    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        print(f"    Validating model {model_name}...")

        # Test with dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 1024, 1024).to(device)

            # Warm-up run
            output = model(dummy_input)

            # Handle different output formats
            if isinstance(output, tuple):
                output = output[0]

            # Check output shape and values
            expected_shape = (1, 1, 1024, 1024)
            if output.shape != expected_shape:
                print(f"    ❌ Invalid output shape: {output.shape}, expected: {expected_shape}")
                return False

            # Check for NaN or Inf values
            if torch.isnan(output).any():
                print(f"    ❌ Output contains NaN values")
                return False

            if torch.isinf(output).any():
                print(f"    ❌ Output contains Inf values")
                return False

            # Check output range (should be reasonable for segmentation)
            output_min, output_max = output.min().item(), output.max().item()
            print(f"    Output range: [{output_min:.3f}, {output_max:.3f}]")

            # Test with sigmoid (typical for segmentation)
            sigmoid_output = torch.sigmoid(output)
            sigmoid_min, sigmoid_max = sigmoid_output.min().item(), sigmoid_output.max().item()

            if not (0 <= sigmoid_min <= 1 and 0 <= sigmoid_max <= 1):
                print(f"    ❌ Sigmoid output out of range: [{sigmoid_min:.3f}, {sigmoid_max:.3f}]")
                return False

            print(f"    ✅ Model validation passed")
            print(f"    Output shape: {output.shape}")
            print(f"    Sigmoid range: [{sigmoid_min:.3f}, {sigmoid_max:.3f}]")

            return True

    except Exception as e:
        print(f"    ❌ Model validation failed: {e}")
        return False

def validate_model_segmentation(model, device, model_name):
    """
    Test that the model can perform actual segmentation on test data

    Args:
        model: The loaded model
        device: Device to run test on
        model_name: Name of the model for logging

    Returns:
        bool: True if segmentation test passes, False otherwise
    """
    try:
        print(f"      Testing segmentation on real data...")

        # Create a realistic test image (similar to actual data)
        with torch.no_grad():
            # Test with a more realistic input pattern
            test_input = torch.randn(1, 3, 1024, 1024).to(device)
            # Normalize to typical image range
            test_input = (test_input - test_input.min()) / (test_input.max() - test_input.min())

            # Forward pass
            output = model(test_input)

            # Handle different output formats
            if isinstance(output, tuple):
                output = output[0]

            # Apply sigmoid for segmentation
            segmentation = torch.sigmoid(output)

            # Check segmentation properties
            seg_min, seg_max = segmentation.min().item(), segmentation.max().item()
            seg_mean = segmentation.mean().item()

            # Validate segmentation output
            if not (0 <= seg_min <= 1 and 0 <= seg_max <= 1):
                print(f"      ❌ Invalid segmentation range: [{seg_min:.3f}, {seg_max:.3f}]")
                return False

            # Check that the model produces varied outputs (not all zeros or ones)
            if seg_max - seg_min < 0.01:
                print(f"      ❌ Segmentation output too uniform: range={seg_max - seg_min:.6f}")
                return False

            # Test with threshold
            binary_seg = (segmentation > 0.5).float()
            positive_ratio = binary_seg.mean().item()

            print(f"      ✅ Segmentation test passed")
            print(f"      Segmentation range: [{seg_min:.3f}, {seg_max:.3f}]")
            print(f"      Mean activation: {seg_mean:.3f}")
            print(f"      Positive ratio (>0.5): {positive_ratio:.3f}")

            return True

    except Exception as e:
        print(f"      ❌ Segmentation test failed: {e}")
        return False

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """Vypočítej segmentační metriky"""
    pred_binary = (pred_mask > threshold).astype(np.uint8).flatten()
    true_binary = true_mask.astype(np.uint8).flatten()
    
    # Základní metriky
    accuracy = accuracy_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)
    
    # IoU (Intersection over Union)
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    iou = intersection / union if union > 0 else 0
    
    # Dice coefficient
    dice = 2 * intersection / (pred_binary.sum() + true_binary.sum()) if (pred_binary.sum() + true_binary.sum()) > 0 else 0
    
    # Specificity (True Negative Rate)
    tn = np.logical_and(pred_binary == 0, true_binary == 0).sum()
    fp = np.logical_and(pred_binary == 1, true_binary == 0).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'dice': dice,
        'specificity': specificity
    }

def get_fixed_threshold():
    """Vrátí fixní threshold 0.5 pro spravedlivé porovnání všech modelů"""
    threshold = 0.5
    print(f"Používám fixní threshold: {threshold} (stejný pro všechny modely)")
    return threshold, []



def evaluate_model(model, dataloader, device, model_name):
    """Evaluuj model na celém datasetu s optimalizovaným měřením rychlosti

    Args:
        model: Model k evaluaci
        dataloader: DataLoader s testovacími daty
        device: Zařízení (cuda/cpu)
        model_name: Jméno modelu
    """
    all_metrics = []
    inference_times = []

    print(f"Evaluuji model {model_name}...")

    # Optimalizované warm-up pro stabilní měření času
    print("  Provádím rozšířený warm-up (30 iterací)...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 1024, 1024).to(device, non_blocking=True)

        # Warm-up 30 iterací pro stabilizaci GPU
        for i in range(30):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if (i + 1) % 10 == 0:
                print(f"    Warm-up: {i + 1}/30")

    print("  Začínám měření rychlosti...")

    # Separátní měření rychlosti na 100 iteracích pro přesnější statistiky
    speed_measurement_times = []
    print("  Měřím rychlost inference (100 iterací)...")

    with torch.no_grad():
        # Použij první obrázek z datasetu pro realistické měření
        first_batch = next(iter(dataloader))
        test_image = first_batch[0][:1].to(device, non_blocking=True)  # Jen první obrázek

        # Stabilizace před měřením
        if device.type == 'cuda':
            torch.cuda.synchronize()

        for i in range(100):
            # Synchronizace před každým měřením
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.perf_counter()  # Přesnější časování
            output = model(test_image)

            # Synchronizace po inference
            if device.type == 'cuda':
                torch.cuda.synchronize()

            inference_time = time.perf_counter() - start_time
            speed_measurement_times.append(inference_time)

            if (i + 1) % 25 == 0:
                print(f"    Rychlost: {i + 1}/100")

    # Statistiky rychlosti
    speed_stats = {
        'mean': np.mean(speed_measurement_times),
        'std': np.std(speed_measurement_times),
        'min': np.min(speed_measurement_times),
        'max': np.max(speed_measurement_times),
        'median': np.median(speed_measurement_times),
        'p95': np.percentile(speed_measurement_times, 95),
        'p99': np.percentile(speed_measurement_times, 99)
    }

    print(f"  Statistiky rychlosti:")
    print(f"    Průměr: {speed_stats['mean']:.4f}s ± {speed_stats['std']:.4f}s")
    print(f"    Medián: {speed_stats['median']:.4f}s")
    print(f"    Min/Max: {speed_stats['min']:.4f}s / {speed_stats['max']:.4f}s")
    print(f"    P95/P99: {speed_stats['p95']:.4f}s / {speed_stats['p99']:.4f}s")

    print(f"  Evaluuji přesnost na celém datasetu...")

    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Pro evaluaci přesnosti použij batch processing (rychlejší)
            # Synchronizace před inferencí
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Single inference pass (no TTA)
            outputs = model(images)

            # Synchronizace po inferenci
            if device.type == 'cuda':
                torch.cuda.synchronize()

            if isinstance(outputs, tuple):  # TransUNet returns tuple
                outputs = outputs[0]

            # Apply sigmoid activation
            outputs = torch.sigmoid(outputs)

            # Převeď na numpy
            pred_masks = outputs.cpu().numpy()
            true_masks = masks.cpu().numpy()

            # Vypočítej metriky pro každý obrázek v batchi s fixním threshold
            for i in range(pred_masks.shape[0]):
                metrics = calculate_metrics(pred_masks[i, 0], true_masks[i], threshold=0.5)
                metrics['filename'] = filenames[i]
                # Použij průměrný čas z měření rychlosti
                metrics['inference_time'] = speed_stats['mean']
                all_metrics.append(metrics)

            if (batch_idx + 1) % 10 == 0:
                print(f"    Přesnost: {batch_idx + 1}/{len(dataloader)} batchů")

    # Přidej statistiky rychlosti do výsledků
    for metric in all_metrics:
        metric['speed_stats'] = speed_stats

    return all_metrics, speed_measurement_times

def save_results_simple(results, output_dir):
    """Ulož výsledky do souborů pro jeden dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset info
    dataset_info = results.get('dataset_info', {})
    dataset_name = dataset_info.get('name', 'unknown')

    # Ulož detailed results jako JSON
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Vytvoř summary tabulku
    summary_data = []
    detailed_data = []

    for model_name, data in results.items():
        if model_name == 'dataset_info':
            continue

        metrics = data['metrics']
        # Extract model type (finetuned/pretrained)
        model_type = "finetuned" if "_finetuned" in model_name else "pretrained" if "_pretrained" in model_name else "unknown"
        base_model_name = model_name.rsplit('_', 1)[0] if ('_finetuned' in model_name or '_pretrained' in model_name) else model_name

        # Získej time_stats pokud existují, jinak použij staré hodnoty
        time_stats = data.get('time_stats', {})

        avg_metrics = {
            'Model': model_name,
            'Base_Model': base_model_name,
            'Model_Type': model_type,
            'Model_Directory': data.get('model_directory', 'N/A'),
            'Model_Timestamp': data.get('model_timestamp', 'N/A'),
            'Dataset': dataset_name,
            'Accuracy': np.mean([m['accuracy'] for m in metrics]),
            'Precision': np.mean([m['precision'] for m in metrics]),
            'Recall': np.mean([m['recall'] for m in metrics]),
            'F1_Score': np.mean([m['f1_score'] for m in metrics]),
            'IoU': np.mean([m['iou'] for m in metrics]),
            'Dice': np.mean([m['dice'] for m in metrics]),
            'Specificity': np.mean([m['specificity'] for m in metrics]),
            # Detailní statistiky rychlosti (v ms pro lepší čitelnost)
            'Avg_Inference_Time_ms': time_stats.get('avg_inference_time', np.mean(data['inference_times'])) * 1000,
            'Std_Inference_Time_ms': time_stats.get('std_inference_time', np.std(data['inference_times'])) * 1000,
            'Median_Inference_Time_ms': time_stats.get('median_inference_time', np.median(data['inference_times'])) * 1000,
            'Min_Inference_Time_ms': time_stats.get('min_inference_time', np.min(data['inference_times'])) * 1000,
            'Max_Inference_Time_ms': time_stats.get('max_inference_time', np.max(data['inference_times'])) * 1000,
            'P95_Inference_Time_ms': time_stats.get('p95_inference_time', np.percentile(data['inference_times'], 95)) * 1000,
            'P99_Inference_Time_ms': time_stats.get('p99_inference_time', np.percentile(data['inference_times'], 99)) * 1000,
            'CV_Percent': time_stats.get('cv_percent', (np.std(data['inference_times']) / np.mean(data['inference_times'])) * 100),
            'Measurement_Count': time_stats.get('measurement_count', len(data['inference_times'])),
            'Optimal_Threshold': data['optimal_threshold']
        }
        summary_data.append(avg_metrics)

        # Create detailed per-image results
        for metric in metrics:
            detailed_row = {
                'Model': model_name,
                'Base_Model': base_model_name,
                'Model_Type': model_type,
                'Dataset': dataset_name,
                'Filename': metric['filename'],
                'Accuracy': metric['accuracy'],
                'Precision': metric['precision'],
                'Recall': metric['recall'],
                'F1_Score': metric['f1_score'],
                'IoU': metric['iou'],
                'Dice': metric['dice'],
                'Specificity': metric['specificity'],
                'Inference_Time_ms': metric['inference_time'] * 1000
            }
            detailed_data.append(detailed_row)

    # Ulož summary jako CSV
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / f'{dataset_name}_model_comparison.csv', index=False)

    # Ulož detailed results jako CSV
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(output_dir / f'{dataset_name}_detailed_results.csv', index=False)

    print(f"📁 Results for {dataset_name} saved to: {output_dir}")
    print(f"   • {dataset_name}_model_comparison.csv: Summary metrics per model")
    print(f"   • {dataset_name}_detailed_results.csv: Per-image metrics")
    print(f"   • detailed_results.json: Complete raw data")

    print(f"\n📊 {dataset_name.upper()} DATASET SUMMARY:")
    print(df_summary[['Model', 'Model_Type', 'IoU', 'Dice', 'F1_Score', 'Avg_Inference_Time_ms']].to_string(index=False))

def evaluate_on_dataset(dataset_name, dataset_path, models_base_paths, valid_models, device, output_base_dir):
    """Evaluate all models on a single dataset"""
    print(f"\n{'='*80}")
    print(f"EVALUATING ON DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"Dataset path: {dataset_path}")

    # Setup dataset
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'

    if not images_dir.exists() or not masks_dir.exists():
        print(f"❌ Images or masks directories do not exist in {dataset_path}")
        return None

    # Create dataset v inference módu pro stabilní měření
    dataset = SegmentationDataset(images_dir, masks_dir, inference_mode=True)
    # Optimalizované nastavení DataLoaderu pro stabilní měření rychlosti:
    # - batch_size=1 pro konzistentní měření (fixní velikost batche)
    # - shuffle=False pro reprodukovatelné výsledky
    # - num_workers=0 pro eliminaci I/O variability během měření rychlosti
    # - pin_memory=True pro rychlejší transfer na GPU
    # - persistent_workers=False (default) pro čisté prostředí
    dataloader = DataLoader(
        dataset,
        batch_size=1,           # Fixní batch=1 pro stabilní měření
        shuffle=False,          # Bez shuffle pro reprodukovatelnost
        num_workers=0,          # Bez paralelního načítání pro eliminaci I/O variability
        pin_memory=True,        # Pinned memory pro rychlejší GPU transfer
        drop_last=False         # Zachovej všechny vzorky
    )
    print(f"✅ Dataset loaded: {len(dataset)} images")

    # Initialize results for this dataset
    results = {
        'dataset_info': {
            'name': dataset_name,
            'path': str(dataset_path),
            'num_images': len(dataset),
            'device': str(device)
        }
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION ON {dataset_name.upper()} DATASET")
    print('='*60)

    # Evaluate each valid model
    evaluation_errors = []
    successful_evaluations = 0

    for model_name, model_path, model_info, _ in valid_models:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_name}")
        print(f"Directory: {model_info['directory']}")

        try:
            # Reload model for evaluation (clean state)
            model = load_model(model_name, model_path, device)

            # Additional validation: test segmentation on a single image
            print(f"    Testing segmentation capability...")
            test_passed = validate_model_segmentation(model, device, model_name)
            if not test_passed:
                error_msg = f"Model {model_name} failed segmentation test"
                evaluation_errors.append((model_name, error_msg))
                print(f"❌ CRITICAL: {error_msg}")
                # Clean up and continue to next model
                if device.type == 'cuda':
                    del model
                    torch.cuda.empty_cache()
                continue

        except (FileNotFoundError, ValueError) as e:
            error_msg = f"Config error for {model_name}: {str(e)}"
            evaluation_errors.append((model_name, error_msg))
            print(f"❌ SKIPPING {model_name}: {error_msg}")
            continue

        except Exception as e:
            error_msg = f"Unexpected error loading {model_name}: {str(e)}"
            evaluation_errors.append((model_name, error_msg))
            print(f"❌ SKIPPING {model_name}: {error_msg}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue

        # Use fixed threshold
        optimal_threshold, threshold_data = get_fixed_threshold()
        print(f"Using threshold: {optimal_threshold:.3f}")

        # Evaluate model with fixed threshold
        metrics, inference_times = evaluate_model(model, dataloader, device, model_name)

        # Validate that evaluation produced reasonable results
        if not metrics or len(metrics) == 0:
            error_msg = f"Model {model_name} produced no evaluation metrics"
            evaluation_errors.append((model_name, error_msg))
            print(f"❌ CRITICAL: {error_msg}")
            continue

        # Check for reasonable metric values
        avg_iou = np.mean([m['iou'] for m in metrics])
        if np.isnan(avg_iou) or avg_iou < 0 or avg_iou > 1:
            error_msg = f"Model {model_name} produced invalid IoU values: {avg_iou}"
            evaluation_errors.append((model_name, error_msg))
            print(f"❌ CRITICAL: {error_msg}")
            continue

        # Store results
        results[model_name] = {
            'model_path': str(model_path),
            'model_directory': model_info['directory'],
            'model_timestamp': str(model_info['timestamp']),
            'optimal_threshold': 0.5,
            'threshold_data': [],
            'metrics': metrics,
            'inference_times': inference_times,
            'avg_metrics': {
                'accuracy': np.mean([m['accuracy'] for m in metrics]),
                'precision': np.mean([m['precision'] for m in metrics]),
                'recall': np.mean([m['recall'] for m in metrics]),
                'f1_score': np.mean([m['f1_score'] for m in metrics]),
                'iou': np.mean([m['iou'] for m in metrics]),
                'dice': np.mean([m['dice'] for m in metrics]),
                'specificity': np.mean([m['specificity'] for m in metrics]),
            },
            'time_stats': {
                'avg_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'median_inference_time': np.median(inference_times),
                'p95_inference_time': np.percentile(inference_times, 95),
                'p99_inference_time': np.percentile(inference_times, 99),
                'measurement_count': len(inference_times),
                'cv_percent': (np.std(inference_times) / np.mean(inference_times)) * 100 if np.mean(inference_times) > 0 else 0
            }
        }

        print(f"Average metrics for {model_name}:")
        for metric, value in results[model_name]['avg_metrics'].items():
            print(f"  {metric}: {value:.4f}")

        # Detailní statistiky rychlosti
        time_stats = results[model_name]['time_stats']
        print(f"  Inference time statistics (100 measurements):")
        print(f"    Mean: {time_stats['avg_inference_time']*1000:.2f} ± {time_stats['std_inference_time']*1000:.2f} ms")
        print(f"    Median: {time_stats['median_inference_time']*1000:.2f} ms")
        print(f"    Min/Max: {time_stats['min_inference_time']*1000:.2f} / {time_stats['max_inference_time']*1000:.2f} ms")
        print(f"    P95/P99: {time_stats['p95_inference_time']*1000:.2f} / {time_stats['p99_inference_time']*1000:.2f} ms")
        print(f"    CV: {time_stats['cv_percent']:.1f}% (nižší = stabilnější)")
        print(f"  Min/Max inference time: {np.min(inference_times)*1000:.2f}/{np.max(inference_times)*1000:.2f} ms")
        print(f"  Total test images: {len(metrics)}")

        successful_evaluations += 1
        print(f"✅ Model {model_name} evaluation completed successfully")

        # Clean up GPU memory
        if device.type == 'cuda':
            del model
            torch.cuda.empty_cache()

    # Show evaluation summary for this dataset
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY FOR {dataset_name.upper()}")
    print('='*60)
    print(f"✅ Successfully evaluated: {successful_evaluations} models")
    print(f"❌ Failed evaluations: {len(evaluation_errors)} models")

    if evaluation_errors:
        print(f"\nFailed models:")
        for model_name, error in evaluation_errors:
            print(f"  ❌ {model_name}: {error}")

    if successful_evaluations == 0:
        print(f"❌ CRITICAL ERROR: No models were successfully evaluated on {dataset_name}")
        return None

    # Save results for this dataset
    dataset_output_dir = Path(output_base_dir) / f"{dataset_name}_results"
    save_results_simple(results, dataset_output_dir)

    print(f"✅ Results for {dataset_name} saved to: {dataset_output_dir}")

    return results, successful_evaluations, evaluation_errors

def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate trained models on test datasets')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only test model loading without running full evaluation')
    args = parser.parse_args()

    # Fixed dataset paths for sequential evaluation (absolute paths)
    # Using merged dataset that combines DTS and SpheroSeg test data
    datasets = [
        ("Merged_DTS_SpheroSeg", Path("/home/prusek/SpheroSeg/NN/diplomka/TEST_DATA/Merged"))
    ]

    # Multiple possible model paths to search
    models_base_paths = [
        Path("/home/prusek/SpheroSeg/NN/diplomka/scripts/training/outputs"),
        Path("/home/prusek/SpheroSeg/NN/diplomka/outputs")
    ]

    output_base_dir = f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"{'='*80}")
    print(f"COMPREHENSIVE MODEL EVALUATION SCRIPT")
    print(f"{'='*80}")
    print(f"Mode: {'DRY RUN (model loading test only)' if args.dry_run else 'FULL EVALUATION'}")

    # FAIR COMPARISON MODE - ALWAYS ENABLED
    print(f"⚖️  FAIR COMPARISON MODE ENABLED")
    print(f"   - Single inference passes only (no TTA)")
    print(f"   - Same evaluation settings for all models")
    print(f"   - Fixed threshold 0.5 for all models")
    print(f"   - Sequential evaluation on both test datasets")
    print(f"   - All models loaded from exact training configurations")

    print(f"\nDatasets to evaluate:")
    for dataset_name, dataset_path in datasets:
        print(f"  - {dataset_name}: {dataset_path}")

    print(f"\nModel search paths:")
    for path in models_base_paths:
        print(f"  - {path}")
    if not args.dry_run:
        print(f"Output base directory: {output_base_dir}")

    # Check dataset paths (only for full evaluation)
    if not args.dry_run:
        for dataset_name, dataset_path in datasets:
            if not dataset_path.exists():
                print(f"❌ Dataset path does not exist: {dataset_name} -> {dataset_path}")
                return
            images_dir = dataset_path / 'images'
            masks_dir = dataset_path / 'masks'
            if not images_dir.exists() or not masks_dir.exists():
                print(f"❌ Images or masks directories do not exist in {dataset_name}: {dataset_path}")
                return

    # Check if at least one model path exists
    valid_model_paths = [path for path in models_base_paths if path.exists()]
    if not valid_model_paths:
        print(f"❌ No valid model paths found: {models_base_paths}")
        return

    print(f"✅ Valid model paths: {len(valid_model_paths)}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"DEVICE SETUP")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL DISCOVERY - ALL 16 MODELS")
    print('='*80)

    # Find ALL models in the outputs directory (all 16 models)
    model_configs = []

    # Search through all directories in the model paths
    for base_path in valid_model_paths:
        print(f"\nSearching in: {base_path}")

        for model_dir in base_path.iterdir():
            if not model_dir.is_dir():
                continue

            # Look for best_model.pth in each directory
            checkpoint_path = model_dir / 'best_model.pth'
            if not checkpoint_path.exists():
                print(f"  ⚠️  Skipping {model_dir.name}: no best_model.pth found")
                continue

            # Extract model info
            dir_name = model_dir.name

            # Get modification time or extract timestamp from directory name
            dir_time = None

            # Try to extract timestamp from directory name (format: YYYYMMDD_HHMMSS)
            timestamp_match = re.search(r'(\d{8}_\d{6})', dir_name)
            if timestamp_match:
                try:
                    dir_time = datetime.strptime(timestamp_match.group(1), '%Y%m%d_%H%M%S')
                except ValueError:
                    pass

            # Fallback to file modification time
            if dir_time is None:
                dir_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)

            model_info = {
                'directory': dir_name,
                'timestamp': dir_time,
                'path': str(checkpoint_path)
            }

            # Determine model name based on directory name with proper mapping
            model_name = dir_name

            # Map specific directory patterns to standardized model names
            if 'resunet_cbam' in dir_name:
                if 'finetune' in dir_name:
                    model_name = 'cbam_unet_finetuned'
                elif 'pretrain' in dir_name:
                    model_name = 'cbam_unet_pretrained'
            elif 'lightm_unet' in dir_name:
                if 'finetune' in dir_name:
                    model_name = 'lightm_unet_finetuned'
                elif 'pretrain' in dir_name:
                    model_name = 'lightm_unet_pretrained'
            elif 'resunet_advanced_mini' in dir_name:
                if 'finetuned' in dir_name:
                    model_name = 'resunet_advanced_mini_finetuned'  # Keep mini separate in name but use resunet_advanced model
                elif 'pretrained' in dir_name:
                    model_name = 'resunet_advanced_mini_pretrained'  # Keep mini separate in name but use resunet_advanced model
            elif 'resunet_advanced' in dir_name and 'mini' not in dir_name:
                if 'finetuned' in dir_name:
                    model_name = 'resunet_advanced_finetuned'  # Regular resunet_advanced
                elif 'pretrained' in dir_name:
                    model_name = 'resunet_advanced_pretrained'  # Regular resunet_advanced
            elif 'resunet_small' in dir_name:
                if 'finetune' in dir_name:
                    model_name = 'resunet_small_finetuned'
                elif 'pretrain' in dir_name:
                    model_name = 'resunet_small_pretrained'
            elif 'pspnet_new' in dir_name:
                if 'finetune' in dir_name:
                    model_name = 'pspnet_new_finetuned'
                elif 'pretrain' in dir_name:
                    model_name = 'pspnet_new_pretrained'
            elif 'unet_finetune' in dir_name:
                model_name = 'unet_finetuned'
            elif 'unet_pretrain' in dir_name:
                model_name = 'unet_pretrained'
            # Keep original names for other models (hrnet_*, resunet_advanced_*)

            model_configs.append((model_name, checkpoint_path, model_info))
            print(f"  ✅ Found: {model_name} -> {dir_name}")

    if not model_configs:
        print("❌ No models found in any search path!")
        return

    # Sort by timestamp (newest first) for consistent ordering
    model_configs.sort(key=lambda x: x[2]['timestamp'], reverse=True)

    print(f"\n✅ TOTAL DISCOVERED: {len(model_configs)} models")
    print(f"📊 Target: 16 models (all trained models)")

    if len(model_configs) < 16:
        print(f"⚠️  WARNING: Expected 16 models but found only {len(model_configs)}")
        print(f"   This may indicate missing models or different directory structure")
    elif len(model_configs) > 16:
        print(f"ℹ️  INFO: Found {len(model_configs)} models (more than expected 16)")
        print(f"   This may include additional experimental models")

    # Print summary of found models
    print(f"\nDISCOVERED MODELS:")
    for i, (model_name, model_path, model_info) in enumerate(model_configs, 1):
        model_type = "finetuned" if "_finetuned" in model_name else "pretrained" if "_pretrained" in model_name else "unknown"
        print(f"  {i:2d}. {model_name} ({model_type}): {model_info['directory']}")

    print(f"\n{'='*80}")
    print("MODEL VALIDATION PHASE")
    print('='*80)

    # Validate each model before evaluation
    valid_models = []
    failed_models = []

    for model_name, model_path, model_info in model_configs:
        print(f"\n{'='*50}")
        print(f"Validating model: {model_name}")
        print(f"Directory: {model_info['directory']}")
        print(f"Path: {model_path}")
        print(f"Timestamp: {model_info['timestamp']}")

        try:
            # Load model using exact config
            model = load_model(model_name, model_path, device)

            # Validate model can perform forward pass
            if validate_model_loading(model, device, model_name):
                valid_models.append((model_name, model_path, model_info, model))
                print(f"✅ Model {model_name} validation PASSED")
            else:
                failed_models.append((model_name, "Forward pass validation failed"))
                print(f"❌ Model {model_name} validation FAILED")

            # Clean up GPU memory
            if device.type == 'cuda':
                del model
                torch.cuda.empty_cache()

        except (FileNotFoundError, ValueError) as e:
            failed_models.append((model_name, f"Config error: {str(e)}"))
            print(f"❌ Model {model_name} loading FAILED: {e}")
            print(f"   This indicates missing or invalid config.json")

            # Clean up GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            failed_models.append((model_name, f"Unexpected error: {str(e)}"))
            print(f"❌ Model {model_name} loading FAILED with unexpected error: {e}")

            # Clean up GPU memory on error
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Print validation summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print('='*80)
    print(f"✅ Successfully validated: {len(valid_models)} models")
    print(f"❌ Failed validation: {len(failed_models)} models")

    if valid_models:
        print(f"\nValid models:")
        for i, (model_name, _, model_info, _) in enumerate(valid_models, 1):
            print(f"  {i:2d}. ✅ {model_name}: {model_info['directory']}")

    if failed_models:
        print(f"\nFailed models:")
        for model_name, error in failed_models:
            print(f"  ❌ {model_name}: {error}")

    # If dry run, stop here
    if args.dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN COMPLETE")
        print('='*80)
        print(f"Model validation completed. {len(valid_models)} models are ready for evaluation.")
        if len(failed_models) > 0:
            print(f"\n⚠️  {len(failed_models)} models failed validation but evaluation can continue with valid models.")
        if len(valid_models) > 0:
            print("\nTo run full evaluation, execute:")
            print("python evaluate_models_server.py")
        else:
            print("❌ No models passed validation!")
            print("❌ Cannot proceed with evaluation - no models loaded successfully")
        return

    # Continue with evaluation - try to evaluate as many models as possible
    if not valid_models:
        print("❌ No valid models found for evaluation!")
        print("❌ Cannot proceed with evaluation - no models loaded successfully")
        return

    # SEQUENTIAL DATASET EVALUATION
    print(f"\n{'='*80}")
    print("SEQUENTIAL DATASET EVALUATION")
    print('='*80)
    print(f"Evaluating {len(valid_models)} models on {len(datasets)} datasets sequentially")
    print(f"Fair comparison mode: Single inference only, fixed threshold 0.5")

    # Store results for each dataset
    all_dataset_results = {}
    overall_summary = {
        'total_models_evaluated': 0,
        'total_datasets': len(datasets),
        'successful_evaluations_per_dataset': {},
        'failed_evaluations_per_dataset': {},
        'evaluation_timestamp': time.strftime('%Y%m%d_%H%M%S')
    }

    # Evaluate on each dataset sequentially
    for dataset_name, dataset_path in datasets:
        print(f"\n{'='*100}")
        print(f"STARTING EVALUATION ON DATASET: {dataset_name.upper()}")
        print(f"{'='*100}")

        # Evaluate all models on this dataset
        dataset_results = evaluate_on_dataset(
            dataset_name, dataset_path, models_base_paths,
            valid_models, device, output_base_dir
        )

        if dataset_results is not None:
            results, successful_count, errors = dataset_results
            all_dataset_results[dataset_name] = results
            overall_summary['successful_evaluations_per_dataset'][dataset_name] = successful_count
            overall_summary['failed_evaluations_per_dataset'][dataset_name] = len(errors)
            overall_summary['total_models_evaluated'] = max(overall_summary['total_models_evaluated'], successful_count)

            print(f"\n✅ {dataset_name} evaluation completed: {successful_count} models successful")
        else:
            print(f"\n❌ {dataset_name} evaluation failed completely")
            overall_summary['successful_evaluations_per_dataset'][dataset_name] = 0
            overall_summary['failed_evaluations_per_dataset'][dataset_name] = len(valid_models)

    # Create combined summary tables
    print(f"\n{'='*100}")
    print("CREATING COMBINED SUMMARY TABLES")
    print('='*100)

    # Create combined results directory
    combined_output_dir = Path(output_base_dir) / "combined_results"
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate combined summary for both datasets
    if all_dataset_results:
        create_combined_summary_tables(all_dataset_results, combined_output_dir)

    # Save overall summary
    summary_file = combined_output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*100}")
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print('='*100)

    total_successful = sum(overall_summary['successful_evaluations_per_dataset'].values())
    total_failed = sum(overall_summary['failed_evaluations_per_dataset'].values())

    print(f"📊 FINAL STATISTICS:")
    print(f"   • Total models discovered: {len(model_configs)}")
    print(f"   • Models passed validation: {len(valid_models)}")
    print(f"   • Datasets evaluated: {len(datasets)}")
    print(f"   • Total successful evaluations: {total_successful}")
    print(f"   • Total failed evaluations: {total_failed}")

    print(f"\n📁 RESULTS STRUCTURE:")
    print(f"   • Base directory: {output_base_dir}")
    for dataset_name in [d[0] for d in datasets]:
        if dataset_name in overall_summary['successful_evaluations_per_dataset']:
            success_count = overall_summary['successful_evaluations_per_dataset'][dataset_name]
            print(f"   • {dataset_name}_results/: {success_count} models evaluated")
    print(f"   • combined_results/: Summary tables and comparisons")

    print(f"\n📋 DATASET-SPECIFIC RESULTS:")
    for dataset_name in [d[0] for d in datasets]:
        if dataset_name in overall_summary['successful_evaluations_per_dataset']:
            success = overall_summary['successful_evaluations_per_dataset'][dataset_name]
            failed = overall_summary['failed_evaluations_per_dataset'][dataset_name]
            print(f"   • {dataset_name}: ✅ {success} successful, ❌ {failed} failed")

    if total_successful > 0:
        print(f"\n✅ EVALUATION COMPLETED SUCCESSFULLY")
        print(f"   Fair comparison ensured: Single inference, fixed threshold, exact training configs")
        print(f"   Results available in separate directories for each dataset")
        print(f"   Combined summary tables created for easy comparison")
    else:
        print(f"\n❌ EVALUATION FAILED")
        print(f"   No models were successfully evaluated on any dataset")

def create_combined_summary_tables(all_dataset_results, output_dir):
    """Create combined summary tables comparing results across datasets"""
    print(f"Creating combined summary tables...")

    # Create a comprehensive comparison table
    combined_data = []

    for dataset_name, results in all_dataset_results.items():
        for model_name, data in results.items():
            if model_name == 'dataset_info':
                continue

            metrics = data['avg_metrics']
            time_stats = data.get('time_stats', {})

            # Extract model type and base name
            model_type = "finetuned" if "_finetuned" in model_name else "pretrained" if "_pretrained" in model_name else "unknown"
            base_model_name = model_name.rsplit('_', 1)[0] if ('_finetuned' in model_name or '_pretrained' in model_name) else model_name

            row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Base_Model': base_model_name,
                'Model_Type': model_type,
                'Model_Directory': data.get('model_directory', 'N/A'),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'IoU': metrics['iou'],
                'Dice': metrics['dice'],
                'Specificity': metrics['specificity'],
                'Avg_Inference_Time_ms': time_stats.get('avg_inference_time', 0) * 1000,
                'Std_Inference_Time_ms': time_stats.get('std_inference_time', 0) * 1000,
                'Median_Inference_Time_ms': time_stats.get('median_inference_time', 0) * 1000,
            }
            combined_data.append(row)

    # Save combined results
    df_combined = pd.DataFrame(combined_data)
    df_combined.to_csv(output_dir / 'combined_model_comparison.csv', index=False)

    # Create dataset-specific summary tables
    for dataset_name in all_dataset_results.keys():
        dataset_data = [row for row in combined_data if row['Dataset'] == dataset_name]
        if dataset_data:
            df_dataset = pd.DataFrame(dataset_data)
            # Remove Dataset column since it's redundant
            df_dataset = df_dataset.drop('Dataset', axis=1)
            df_dataset.to_csv(output_dir / f'{dataset_name}_summary.csv', index=False)

    print(f"✅ Combined summary tables saved to: {output_dir}")
    print(f"   • combined_model_comparison.csv: All results in one table")
    for dataset_name in all_dataset_results.keys():
        print(f"   • {dataset_name}_summary.csv: {dataset_name} dataset results only")

if __name__ == '__main__':
    main()
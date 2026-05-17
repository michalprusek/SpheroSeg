# CNN_main_spheroid.py - Modified for spheroid dataset with multi-GPU support

import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import torchvision
import logging
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pickle
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Import model architectures
from models.hrnet import HRNetV2
from models.pspnet import PSPNet
from models.unet import UNet
from models.resunet_cbam import ResUNetCBAM
from models.lightm_unet import LightMUNet
# Import MA and LC variants
from models.resunet_ma import AdvancedResUNet as MAResUNet
from models.resunet_ma_mini import AdvancedResUNet as MAMiniResUNet
from models.resunet_lc import ResUNetSmall as LCResUNet

# ===========================
# Dataset Class for Spheroid
# ===========================
class CachedSpheroidDataset(Dataset):
    """Dataset with optional caching for faster loading"""
    def __init__(self, dataset_dir, split='train', transform=None, use_cache=True, cache_dir=None):
        """
        Args:
            dataset_dir: Path to training_big directory
            split: 'train', 'val', or 'test'
            transform: Albumentations transformations
            use_cache: Whether to cache loaded images
            cache_dir: Directory for cache files (default: dataset_dir/.cache)
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.use_cache = use_cache
        
        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = self.dataset_dir / '.cache' / split
        else:
            self.cache_dir = Path(cache_dir) / split
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup directories
        self.split_dir = self.dataset_dir / split
        self.image_dir = self.split_dir / 'images'
        self.mask_dir = self.split_dir / 'masks'
        
        # Supported image formats
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        
        # Get all image files
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend([f for f in self.image_dir.glob(ext) if not f.name.startswith('._')])
            # Also check uppercase extensions
            self.image_files.extend([f for f in self.image_dir.glob(ext.upper()) if not f.name.startswith('._')])
        
        self.image_files = sorted(list(set(self.image_files)))  # Remove duplicates and sort
        
        # Create mask lookup dictionary for faster matching
        mask_files = []
        for ext in image_extensions:
            mask_files.extend([f for f in self.mask_dir.glob(ext) if not f.name.startswith('._')])
            mask_files.extend([f for f in self.mask_dir.glob(ext.upper()) if not f.name.startswith('._')])
        
        # Create lookup by stem and by name without common suffixes
        mask_lookup = {}
        for mask_path in mask_files:
            # Add by stem
            mask_lookup[mask_path.stem] = mask_path
            # Add by name without extension
            mask_lookup[mask_path.name.rsplit('.', 1)[0]] = mask_path
            # Special handling for .ome.tiff -> .tiff matching
            if mask_path.name.endswith('.tiff') or mask_path.name.endswith('.tif'):
                # Also register without .ome if present
                clean_name = mask_path.name.replace('.ome.tiff', '').replace('.ome.tif', '')
                clean_name = clean_name.replace('.ome.TIFF', '').replace('.ome.TIF', '')
                clean_name = clean_name.replace('.tiff', '').replace('.tif', '')
                clean_name = clean_name.replace('.TIFF', '').replace('.TIF', '')
                mask_lookup[clean_name] = mask_path
        
        # Validate that masks exist
        self.valid_files = []
        missing_masks = []
        for img_path in self.image_files:
            mask_found = False
            
            # Try different matching strategies
            # 1. Try exact stem match
            if img_path.stem in mask_lookup:
                self.valid_files.append((img_path, mask_lookup[img_path.stem]))
                mask_found = True
            # 2. Try name without extension
            elif img_path.name.rsplit('.', 1)[0] in mask_lookup:
                self.valid_files.append((img_path, mask_lookup[img_path.name.rsplit('.', 1)[0]]))
                mask_found = True
            # 3. Special handling for .ome.tiff images
            elif '.ome.' in img_path.name:
                # Remove .ome.tiff/.ome.tif suffix
                clean_name = img_path.name
                for suffix in ['.ome.tiff', '.ome.tif', '.ome.TIFF', '.ome.TIF']:
                    clean_name = clean_name.replace(suffix, '')
                if clean_name in mask_lookup:
                    self.valid_files.append((img_path, mask_lookup[clean_name]))
                    mask_found = True
            
            if not mask_found:
                missing_masks.append(img_path.name)
        
        # Report missing masks
        if missing_masks:
            print(f"\nWarning: Found {len(missing_masks)} images without corresponding masks in {split} set:")
            for i, img in enumerate(missing_masks[:5]):  # Show first 5
                print(f"  - {img}")
            if len(missing_masks) > 5:
                print(f"  ... and {len(missing_masks) - 5} more")
        
        # Final count
        total_images = len(self.image_files)
        valid_pairs = len(self.valid_files)
        print(f"\n{split.upper()} Dataset Summary:")
        print(f"  Total images found: {total_images}")
        print(f"  Valid image-mask pairs: {valid_pairs}")
        print(f"  Images without masks: {total_images - valid_pairs}")
        print(f"  Final {split} dataset size: {valid_pairs}")
    
    def __len__(self):
        return len(self.valid_files)
    
    def _get_cache_path(self, idx):
        """Generate cache file path for given index"""
        img_path, mask_path = self.valid_files[idx]
        # Create hash of full paths for cache key
        cache_key = hashlib.md5(f"{img_path}_{mask_path}".encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_and_cache_item(self, idx):
        """Load item from disk and optionally cache it"""
        img_path, mask_path = self.valid_files[idx]
        # valid_files now contains full paths, not just filenames
        
        # Load image and mask
        image = cv.imread(str(img_path))
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Error loading {img_path.name} or {mask_path.name}")
            
        # Convert BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Ensure binary mask
        mask = (mask > 127).astype(np.float32)
        
        # Cache if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(idx)
            with open(cache_path, 'wb') as f:
                pickle.dump({'image': image, 'mask': mask}, f)
        
        return image, mask
    
    def __getitem__(self, idx):
        # Try to load from cache first
        if self.use_cache:
            cache_path = self._get_cache_path(idx)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    image, mask = data['image'], data['mask']
                except Exception as e:
                    # Cache corrupted, reload from disk
                    print(f"Cache load failed for idx {idx}: {e}")
                    image, mask = self._load_and_cache_item(idx)
            else:
                image, mask = self._load_and_cache_item(idx)
        else:
            image, mask = self._load_and_cache_item(idx)
        
        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask

# Keep original SpheroidDataset for backward compatibility
class SpheroidDataset(CachedSpheroidDataset):
    def __init__(self, dataset_dir, split='train', transform=None):
        super().__init__(dataset_dir, split, transform, use_cache=False)

# ===========================
# Loss Functions
# ===========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return F_loss.mean()
        else:
            return F_loss

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        iou = (intersection + self.eps) / (union + self.eps)

        return 1 - iou

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
        return 1 - dice

class BoundaryLoss(nn.Module):
    """Boundary loss for better edge detection in segmentation"""
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def compute_dtm(self, img_gt, out_shape):
        """
        Compute distance transform map
        img_gt: ground truth binary mask
        """
        fg_dtm = torch.zeros_like(img_gt).float()
        bg_dtm = torch.zeros_like(img_gt).float()
        
        for b in range(img_gt.shape[0]):
            mask = img_gt[b, 0].cpu().numpy()
            # Distance transform for foreground and background
            import cv2
            import numpy as np
            
            if mask.max() > 0:
                fg_mask = mask.astype(np.uint8)
                bg_mask = 1 - fg_mask
                
                fg_dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                bg_dist = cv2.distanceTransform(bg_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                
                fg_dtm[b, 0] = torch.from_numpy(fg_dist).to(img_gt.device)
                bg_dtm[b, 0] = torch.from_numpy(bg_dist).to(img_gt.device)
        
        return fg_dtm, bg_dtm
    
    def forward(self, logits, targets):
        """
        logits: network output
        targets: ground truth masks
        """
        preds = torch.sigmoid(logits)
        
        # Compute distance transform maps
        fg_dtm, bg_dtm = self.compute_dtm(targets, preds.shape)
        
        # Compute boundary loss
        dtm = fg_dtm - bg_dtm
        dtm[dtm > self.theta0] = self.theta0
        dtm[dtm < -self.theta0] = -self.theta0
        
        e_psi = 1e-6 + preds * (1 - 2 * targets)
        boundary_loss = torch.mean(dtm * e_psi)
        
        return boundary_loss

class CombinedLoss(nn.Module):
    """Combined loss with Focal, Dice, IoU and optional Boundary components"""
    def __init__(self, focal_weight=1.0, dice_weight=1.0, iou_weight=0.5, boundary_weight=0.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=0.8, gamma=2)
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.boundary = BoundaryLoss() if boundary_weight > 0 else None
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        iou_loss = self.iou(logits, targets)
        
        total_loss = (self.focal_weight * focal_loss + 
                     self.dice_weight * dice_loss + 
                     self.iou_weight * iou_loss)
        
        components = {
            'focal': focal_loss.item(),
            'dice': dice_loss.item(),
            'iou': iou_loss.item()
        }
        
        if self.boundary_weight > 0 and self.boundary is not None:
            boundary_loss = self.boundary(logits, targets)
            total_loss += self.boundary_weight * boundary_loss
            components['boundary'] = boundary_loss.item()
        
        return total_loss, components

# ===========================
# PSPNet Auxiliary Loss Handling
# ===========================
def handle_model_output(model, images, criterion, masks, aux_weight=0.4):
    """
    Handle model output for both single output models and PSPNet with auxiliary output.

    Args:
        model: The neural network model
        images: Input images tensor
        criterion: Loss function
        masks: Target masks tensor
        aux_weight: Weight for auxiliary loss (default 0.4)

    Returns:
        tuple: (main_output, total_loss, loss_components)
    """
    outputs = model(images)

    # Check if model returned dual outputs (PSPNet in training mode)
    if isinstance(outputs, tuple) and len(outputs) == 2:
        # PSPNet with auxiliary output
        main_output, aux_output = outputs

        # Calculate losses for both outputs
        main_loss, main_components = criterion(main_output, masks)
        aux_loss, aux_components = criterion(aux_output, masks)

        # Combine losses
        total_loss = main_loss + aux_weight * aux_loss

        # Combine loss components for logging
        combined_components = {}
        for key in main_components:
            combined_components[key] = main_components[key]
            if key in aux_components:
                combined_components[f'aux_{key}'] = aux_components[key]
        combined_components['aux_weight'] = aux_weight
        combined_components['main_loss'] = main_loss.item()
        combined_components['aux_loss'] = aux_loss.item()

        return main_output, total_loss, combined_components
    else:
        # Single output model (ResUNet, HRNet, etc.)
        main_loss, components = criterion(outputs, masks)
        return outputs, main_loss, components

def is_pspnet_model(model):
    """Check if the model is PSPNet (new architecture)"""
    # Check if it's wrapped in DataParallel or DistributedDataParallel
    actual_model = model.module if hasattr(model, 'module') else model

    # Check if it's the new PSPNet architecture
    return (hasattr(actual_model, 'ppm') and
            hasattr(actual_model, 'aux') and
            hasattr(actual_model, 'cls'))

# ===========================
# Metrics
# ===========================
def calculate_metrics(preds, targets, threshold=0.5):
    """Calculate IoU, Dice, Precision, Recall, F1"""
    with torch.no_grad():
        # Threshold predictions
        preds = (preds > threshold).float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate metrics
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2. * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)
        
        # Precision, Recall, F1
        tp = intersection
        fp = preds.sum() - intersection
        fn = targets.sum() - intersection
        
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'iou': iou.item(),
            'dice': dice.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }

# ===========================
# Training Functions
# ===========================
def train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, writer, rank=0, scheduler=None, gradient_accumulation_steps=1, gradient_clip_val=1.0, aux_weight=0.4):
    model.train()
    running_loss = 0.0
    running_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    loss_components = {'focal': 0, 'dice': 0, 'iou': 0, 'boundary': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training', disable=(rank != 0))
    
    # Zero gradients at the start
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            outputs, loss, components = handle_model_output(model, images, criterion, masks, aux_weight)
            
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Check for NaN in loss BEFORE backward
        if torch.isnan(loss).any():
            print(f"NaN detected in loss: {loss}, skipping batch")
            continue
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Perform optimizer step only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Enhanced gradient clipping with stability checks
            if gradient_clip_val > 0:
                scaler.unscale_(optimizer)

                # Check for NaN/inf gradients before clipping
                has_nan_inf = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_inf = True
                            break

                if has_nan_inf:
                    print(f"NaN/Inf gradients detected at batch {batch_idx}, skipping optimizer step")
                    scaler.update()  # Update scaler but skip optimizer step
                    optimizer.zero_grad()
                    continue

                # Apply gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)

                # More conservative gradient norm monitoring
                if total_norm > gradient_clip_val * 2:  # Reduced threshold from 5 to 2
                    if total_norm > gradient_clip_val * 10:  # Very large gradients
                        print(f"WARNING: Very large gradient norm: {total_norm:.4f}, consider reducing LR")
                    elif batch_idx % 50 == 0:  # Log less frequently
                        print(f"Large gradient norm: {total_norm:.4f}, clipping applied")

            # Check if scaler is ready for step
            if scaler.get_scale() > 0:
                scaler.step(optimizer)
                scaler.update()
            else:
                print("Scaler scale is 0, skipping optimizer step")
                scaler.update()

            optimizer.zero_grad()
            
            # Update OneCycleLR scheduler per optimizer step (not per batch)
            if scheduler is not None and hasattr(scheduler, 'step') and scheduler.__class__.__name__ == 'OneCycleLR':
                scheduler.step()
        
        # Update metrics (with unscaled loss for accurate tracking)
        running_loss += loss.item() * gradient_accumulation_steps
        metrics = calculate_metrics(torch.sigmoid(outputs), masks)
        for k, v in metrics.items():
            running_metrics[k] += v
        for k, v in components.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v
            
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'loss': loss.item() * gradient_accumulation_steps,
                'iou': metrics['iou'],
                'dice': metrics['dice']
            })
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/BatchLoss', loss.item() * gradient_accumulation_steps, global_step)
                writer.add_scalar('Train/BatchIoU', metrics['iou'], global_step)

    # Handle remaining gradients at the end of epoch
    # This ensures that any accumulated gradients from the last incomplete batch are applied
    if len(train_loader) % gradient_accumulation_steps != 0:
        if rank == 0:
            print(f"Applying remaining gradients at end of epoch (last {len(train_loader) % gradient_accumulation_steps} batches)")

        # Enhanced gradient clipping with stability checks
        if gradient_clip_val > 0:
            scaler.unscale_(optimizer)

            # Check for NaN/inf gradients before clipping
            has_nan_inf = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_inf = True
                        break

            if not has_nan_inf:
                # Apply gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)

                # Check if scaler is ready for step
                if scaler.get_scale() > 0:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scaler.update()
            else:
                print("NaN/Inf gradients detected in final step, skipping")
                scaler.update()
        else:
            # No gradient clipping
            if scaler.get_scale() > 0:
                scaler.step(optimizer)
                scaler.update()
            else:
                scaler.update()

        optimizer.zero_grad()

    # Calculate epoch averages
    n_batches = len(train_loader)
    avg_loss = running_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    avg_components = {k: v / n_batches for k, v in loss_components.items()}

    return avg_loss, avg_metrics, avg_components

# ===========================
# Test Time Augmentation
# ===========================
def apply_tta(model, images, device):
    """Apply Test Time Augmentation for better predictions"""
    model.eval()
    batch_size, _, h, w = images.shape
    
    # List to store predictions
    predictions = []
    
    with torch.no_grad():
        # Original
        pred1 = torch.sigmoid(model(images))
        predictions.append(pred1)
        
        # Horizontal flip
        images_hflip = torch.flip(images, dims=[3])
        pred2 = torch.sigmoid(model(images_hflip))
        pred2 = torch.flip(pred2, dims=[3])
        predictions.append(pred2)
        
        # Vertical flip
        images_vflip = torch.flip(images, dims=[2])
        pred3 = torch.sigmoid(model(images_vflip))
        pred3 = torch.flip(pred3, dims=[2])
        predictions.append(pred3)
        
        # 90 degree rotation
        images_rot90 = torch.rot90(images, k=1, dims=[2, 3])
        pred4 = torch.sigmoid(model(images_rot90))
        pred4 = torch.rot90(pred4, k=-1, dims=[2, 3])
        predictions.append(pred4)
        
        # 180 degree rotation
        images_rot180 = torch.rot90(images, k=2, dims=[2, 3])
        pred5 = torch.sigmoid(model(images_rot180))
        pred5 = torch.rot90(pred5, k=-2, dims=[2, 3])
        predictions.append(pred5)
        
        # 270 degree rotation
        images_rot270 = torch.rot90(images, k=3, dims=[2, 3])
        pred6 = torch.sigmoid(model(images_rot270))
        pred6 = torch.rot90(pred6, k=-3, dims=[2, 3])
        predictions.append(pred6)
        
        # Diagonal flip (transpose)
        images_transpose = images.transpose(2, 3)
        pred7 = torch.sigmoid(model(images_transpose))
        pred7 = pred7.transpose(2, 3)
        predictions.append(pred7)
        
        # Anti-diagonal flip
        images_antidiag = torch.flip(images.transpose(2, 3), dims=[2])
        pred8 = torch.sigmoid(model(images_antidiag))
        pred8 = torch.flip(pred8, dims=[2]).transpose(2, 3)
        predictions.append(pred8)
    
    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred

def validate_epoch(model, val_loader, criterion, device, epoch, writer, rank=0, use_tta=False, aux_weight=0.4):
    model.eval()
    running_loss = 0.0
    running_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} Validation', disable=(rank != 0))
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.float().unsqueeze(1).to(device)
            
            if use_tta:
                preds = apply_tta(model, images, device)
                # For TTA, we already have probabilities, calculate loss directly
                # Avoid double sigmoid by using BCELoss instead of BCEWithLogitsLoss component
                with torch.amp.autocast('cuda'):
                    outputs, loss, _ = handle_model_output(model, images, criterion, masks, aux_weight)
            else:
                outputs, loss, _ = handle_model_output(model, images, criterion, masks, aux_weight)
                preds = torch.sigmoid(outputs)
            running_loss += loss.item()
            metrics = calculate_metrics(preds, masks)
            for k, v in metrics.items():
                running_metrics[k] += v
                
            if rank == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'iou': metrics['iou'],
                    'dice': metrics['dice']
                })
    
    # Calculate averages
    n_batches = len(val_loader)
    avg_loss = running_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    
    return avg_loss, avg_metrics

# ===========================
# Augmentation Pipeline
# ===========================
def get_training_augmentation(img_size):
    """Strong augmentation for spheroid images"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, p=0.5)
        ], p=0.8),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1)
        ], p=0.9),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1)
        ], p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), 
                       hole_width_range=(16, 32), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_validation_augmentation(img_size):
    """Minimal augmentation for validation"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ===========================
# Multi-GPU Setup
# ===========================
def setup_distributed(rank, world_size):
    """Initialize distributed training with error handling"""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # NCCL environment variables for stability and driver compatibility
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback for localhost
        os.environ['NCCL_P2P_DISABLE'] = '1'     # Disable P2P for stability
        os.environ['NCCL_IB_DISABLE'] = '1'      # Disable InfiniBand
        os.environ['NCCL_NET_GDR_LEVEL'] = '0'   # Disable GPU Direct RDMA
        os.environ['NCCL_NVLS_ENABLE'] = '0'     # Disable NVLS
        os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'  # Ignore P2P issues
        os.environ['NCCL_IGNORE_CPU_AFFINITY'] = '1'  # Ignore CPU affinity
        os.environ['NCCL_NVML_DISABLE'] = '1'    # Disable NVML to avoid version mismatch
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(world_size))

        # Check CUDA availability and driver compatibility
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Try to initialize NCCL backend with timeout
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30)
        )
        torch.cuda.set_device(rank)

        # Test NCCL communication
        test_tensor = torch.ones(1).cuda(rank)
        dist.all_reduce(test_tensor)

        print(f"Successfully initialized distributed training on rank {rank}")
        return True

    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Multi-GPU training failed. Please check CUDA drivers and NCCL installation.")
        raise RuntimeError(f"Distributed training initialization failed: {e}")

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

# ===========================
# Learning Rate Finder
# ===========================
class LRFinder:
    """Learning rate finder to find optimal learning rate"""
    def __init__(self, model, optimizer, criterion, device):
        # Unwrap DDP model if needed to avoid unused parameter issues
        self.model = model.module if hasattr(model, 'module') else model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'lr': [], 'loss': []}
        
    def find(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100, smooth_factor=0.95):
        """Find optimal learning rate"""
        self.model.train()
        lr_schedule = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        
        iterator = iter(train_loader)
        best_loss = float('inf')
        avg_loss = 0
        
        for i, lr in enumerate(tqdm(lr_schedule, desc="Finding LR")):
            # Get batch
            try:
                images, masks = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, masks = next(iterator)
            
            images = images.to(self.device)
            masks = masks.float().unsqueeze(1).to(self.device)
            
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            if isinstance(loss, tuple):
                loss = loss[0]  # For combined loss
            
            # Smooth loss
            if i == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_factor * avg_loss + (1 - smooth_factor) * loss.item()
            
            # Record
            self.history['lr'].append(lr)
            self.history['loss'].append(avg_loss)
            
            # Check if loss is exploding
            if avg_loss > best_loss * 4:
                print(f"\nStopping early, loss is exploding (loss={avg_loss:.4f})")
                break
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        return self.history
    
    def plot(self, save_path=None):
        """Plot learning rate vs loss"""
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.history['lr'], self.history['loss'])
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def suggest_lr(self, method='gradient'):
        """Suggest optimal learning rate"""
        losses = np.array(self.history['loss'])
        lrs = np.array(self.history['lr'])
        
        if method == 'gradient':
            # Find steepest gradient
            gradients = np.gradient(losses)
            min_gradient_idx = np.argmin(gradients)
            suggested_lr = lrs[min_gradient_idx] / 10  # A bit before steepest
        elif method == 'minimum':
            # Find minimum loss
            min_loss_idx = np.argmin(losses)
            suggested_lr = lrs[min_loss_idx] / 10
        else:
            # Default: 1/10 of LR at minimum loss
            min_loss_idx = np.argmin(losses)
            suggested_lr = lrs[min_loss_idx] / 10
            
        return suggested_lr

# ===========================
# Enhanced Early Stopping
# ===========================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=1e-4, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.mode == 'max':
            score = score
        else:
            score = -score
            
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose and score > self.best_score:
                print(f'Validation score improved from {self.best_score:.4f} to {score:.4f}')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop

# ===========================
# Model Selection
# ===========================
def get_model(model_name, num_classes=1, pretrained=True, img_size=1024, use_instance_norm=True, use_checkpoint=False):
    """Get model by name"""
    if model_name.lower() == 'hrnet':
        # HRNet with similar capacity to ResUNet
        return HRNetV2(n_class=num_classes, pretrained=pretrained, use_instance_norm=use_instance_norm)
    elif model_name.lower() == 'pspnet':
        # PSPNet with ResNet101 backbone
        return PSPNet(n_class=num_classes, backbone='resnet101', pretrained=pretrained, use_instance_norm=use_instance_norm)
    elif model_name.lower() == 'unet':
        # Basic UNet architecture
        return UNet(in_channels=3, out_channels=num_classes,
                   use_instance_norm=use_instance_norm, dropout_rate=0.1)
    elif model_name.lower() == 'resunet_cbam':
        # ResUNet with full CBAM (channel + spatial attention)
        return ResUNetCBAM(in_channels=3, out_channels=num_classes,
                          use_instance_norm=use_instance_norm, dropout_rate=0.15)
    elif model_name.lower() == 'lightm_unet':
        # LightM-UNet: Mamba-assisted UNet for efficient segmentation
        return LightMUNet(in_channels=3, out_channels=num_classes,
                         base_channels=32, use_instance_norm=use_instance_norm,
                         encoder_layers=[1, 2, 2], decoder_layers=[2, 2, 1],
                         bottleneck_layers=4, dropout_rate=0.1)
    elif model_name.lower() == 'resunet_ma':
        # MA-ResUNet: Multi-Attention ResUNet with SimAM, NAM, and Triplet Attention
        return MAResUNet(in_channels=3, out_channels=num_classes,
                        use_instance_norm=use_instance_norm)
    elif model_name.lower() == 'resunet_ma_mini':
        # MA-Mini-ResUNet: Lightweight version of MA-ResUNet
        return MAMiniResUNet(in_channels=3, out_channels=num_classes,
                            use_instance_norm=use_instance_norm)
    elif model_name.lower() == 'resunet_lc':
        # LC-ResUNet: Lightweight CBAM ResUNet (actually ResUNetSmall in the file)
        return LCResUNet(in_channels=3, out_channels=num_classes,
                        use_instance_norm=use_instance_norm, dropout_rate=0.15)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def freeze_backbone_layers(model, model_name, freeze_ratio=0.5):
    """Freeze backbone layers for finetuning"""
    if model_name.lower() == 'hrnet':
        # Freeze initial layers and stage1
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False
        for param in model.bn2.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        print("Froze initial layers and stage1 in HRNet")
        
    elif model_name.lower() == 'pspnet':
        # Freeze ResNet backbone layers
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        print("Froze layer0, layer1, and layer2 in PSPNet backbone")
        
    elif model_name.lower() in ['unet', 'resunet_cbam', 'resunet_lc']:
        # Freeze encoder blocks for new architectures
        if hasattr(model, 'downs'):
            modules = list(model.downs.children()) if hasattr(model.downs, 'children') else model.downs
            freeze_count = int(len(modules) * freeze_ratio)
            for i, module in enumerate(list(modules)[:freeze_count]):
                for param in module.parameters():
                    param.requires_grad = False
            print(f"Froze {freeze_count}/{len(modules)} encoder blocks in {model_name}")
        elif hasattr(model, 'encoder_blocks'):
            # For UNet variants with encoder_blocks
            modules = model.encoder_blocks
            freeze_count = int(len(modules) * freeze_ratio)
            for i, module in enumerate(list(modules)[:freeze_count]):
                for param in module.parameters():
                    param.requires_grad = False
            print(f"Froze {freeze_count}/{len(modules)} encoder blocks in {model_name}")

    elif model_name.lower() in ['resunet_ma', 'resunet_ma_mini']:
        # Freeze encoder blocks for MA-ResUNet variants
        encoders = [model.encoder1, model.encoder2, model.encoder3, model.encoder4]
        freeze_count = int(len(encoders) * freeze_ratio)
        for i, encoder in enumerate(encoders[:freeze_count]):
            for param in encoder.parameters():
                param.requires_grad = False
        print(f"Froze {freeze_count}/{len(encoders)} encoder blocks in {model_name}")
    
    elif model_name.lower() == 'lightm_unet':
        # Freeze encoder blocks for LightM-UNet
        # Freeze initial conv and first two encoder blocks
        for param in model.init_conv.parameters():
            param.requires_grad = False
        for param in model.encoder1.parameters():
            param.requires_grad = False
        for param in model.encoder2.parameters():
            param.requires_grad = False
        print("Froze initial conv and first 2 encoder blocks in LightM-UNet")
    
    return model

def unfreeze_all_layers(model):
    """Unfreeze all layers"""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfroze all layers")

# ===========================
# Main Training Function
# ===========================
def train(rank, world_size, args):
    # Setup distributed training if using multiple GPUs
    if world_size > 1:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        print(f"Successfully initialized distributed training on rank {rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging and tensorboard (only on rank 0)
    if rank == 0:
        # Logger
        logger = logging.getLogger('SpheroidSegmentation')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(output_dir / 'training.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
        
        # Tensorboard
        writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
        
        # Save configuration
        config = vars(args)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    else:
        writer = None
        logger = None
    
    # Create datasets with optional caching
    train_dataset = CachedSpheroidDataset(
        args.dataset_path, 
        split='train',
        transform=get_training_augmentation(args.img_size),
        use_cache=args.use_cache
    )
    
    val_dataset = CachedSpheroidDataset(
        args.dataset_path,
        split='val', 
        transform=get_validation_augmentation(args.img_size),
        use_cache=args.use_cache
    )
    
    # Also load test dataset to show complete statistics
    if rank == 0:
        test_dataset = CachedSpheroidDataset(
            args.dataset_path,
            split='test',
            transform=get_validation_augmentation(args.img_size),
            use_cache=args.use_cache
        )
        
        # Print complete dataset summary
        print("\n" + "="*60)
        print("COMPLETE DATASET SUMMARY")
        print("="*60)
        print(f"Dataset path: {args.dataset_path}")
        print(f"Image size: {args.img_size}x{args.img_size}")
        print("-"*60)
        print(f"Train samples: {len(train_dataset):,}")
        print(f"Val samples:   {len(val_dataset):,}")
        print(f"Test samples:  {len(test_dataset):,}")
        print("-"*60)
        print(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
        print("="*60 + "\n")
        
        # Log to file
        if logger:
            logger.info("="*60)
            logger.info("COMPLETE DATASET SUMMARY")
            logger.info("="*60)
            logger.info(f"Train samples: {len(train_dataset):,}")
            logger.info(f"Val samples:   {len(val_dataset):,}")
            logger.info(f"Test samples:  {len(test_dataset):,}")
            logger.info(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
            logger.info("="*60)
    
    # Create data loaders
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = get_model(args.model, num_classes=1, pretrained=args.pretrained, 
                     img_size=args.img_size, use_instance_norm=args.use_instance_norm,
                     use_checkpoint=args.use_checkpoint)
    
    # Load pretrained weights if provided (for finetuning)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained model from {args.pretrained_path}")

        # Safe checkpoint loading for PyTorch 2.6+
        try:
            # Try with weights_only=True first (safer) with safe globals for argparse.Namespace
            import argparse
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"Warning: weights_only=True with safe globals failed ({e}), trying weights_only=False...")
            try:
                # Fallback to weights_only=False with pickle_module restriction
                import pickle
                checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False,
                                      pickle_module=pickle)
            except Exception as e2:
                print(f"Warning: Standard loading failed ({e2}), trying legacy mode...")
                # Last resort: load without weights_only parameter (older PyTorch compatibility)
                checkpoint = torch.load(args.pretrained_path, map_location=device)
        # Handle both DDP and non-DDP checkpoints
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        # Try to load with strict=True first
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Loaded pretrained model with best IoU: {checkpoint.get('best_iou', 'N/A')}")
        except RuntimeError as e:
            print(f"Warning: Could not load model with strict=True due to architecture mismatch.")
            print("Attempting to load compatible weights only...")
            
            # Manually load compatible weights
            model_dict = model.state_dict()
            compatible_dict = {}
            incompatible_keys = []
            
            for k, v in new_state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        compatible_dict[k] = v
                    else:
                        incompatible_keys.append(f"{k}: {v.shape} -> {model_dict[k].shape}")
            
            # Update model with compatible weights
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict, strict=False)
            
            print(f"\nLoaded {len(compatible_dict)} compatible parameters out of {len(new_state_dict)}")
            print(f"Skipped {len(new_state_dict) - len(compatible_dict)} incompatible parameters")
            
            if incompatible_keys and len(incompatible_keys) <= 20:
                print("\nIncompatible parameters (shape mismatches):")
                for key in incompatible_keys[:20]:
                    print(f"  - {key}")
                if len(incompatible_keys) > 20:
                    print(f"  ... and {len(incompatible_keys) - 20} more")
                    
            print(f"\nNote: The pretrained model appears to be from a different PSPNet variant.")
            print(f"The backbone weights have been loaded successfully.")
            print(f"Loaded pretrained model (partial) with best IoU: {checkpoint.get('best_iou', 'N/A')}")
        
        # Freeze backbone if specified
        if args.freeze_backbone_epochs > 0:
            model = freeze_backbone_layers(model, args.model, freeze_ratio=0.5)
    
    model = model.to(device)
    
    # Wrap model for distributed training
    if world_size > 1:
        # Set find_unused_parameters=True for models with complex architectures
        # that might have unused parameters during certain phases (like LR finding)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Loss function
    criterion = CombinedLoss(
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        iou_weight=args.iou_weight,
        boundary_weight=args.boundary_weight
    )
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Learning Rate Finder (optional)
    if args.find_lr:
        print("\nRunning Learning Rate Finder...")
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_history = lr_finder.find(train_loader, start_lr=1e-7, end_lr=1, num_iter=100)
        
        # Save LR finder plot
        lr_finder.plot(save_path=output_dir / 'lr_finder.png')
        
        # Suggest learning rate
        suggested_lr = lr_finder.suggest_lr(method='gradient')
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        
        if rank == 0:
            logger.info(f"Learning Rate Finder completed. Suggested LR: {suggested_lr:.2e}")
            
        # Reset model and optimizer
        model = get_model(args.model, num_classes=1, pretrained=args.pretrained, 
                         img_size=args.img_size, use_instance_norm=args.use_instance_norm,
                         use_checkpoint=args.use_checkpoint)
        model = model.to(device)
        if world_size > 1:
            # Set find_unused_parameters=True for models with complex architectures
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        # Use suggested LR if not manually set
        if args.lr == 1e-3:  # Default value
            args.lr = suggested_lr
            print(f"Using suggested learning rate: {args.lr:.2e}")
            
        # Recreate optimizer with new LR
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        # Rebind scheduler to the new optimizer — the original was created above
        # against an optimizer that has now been discarded, so scheduler.step()
        # would silently update no-op LR state or raise.
        if args.scheduler == 'reduce':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        elif args.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        elif args.scheduler == 'onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='max', verbose=True)
    
    # Training loop
    best_iou = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # Unfreeze layers after freeze_backbone_epochs
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
            print(f"\nEpoch {epoch}: Unfreezing all layers for full finetuning")
            if world_size > 1:
                unfreeze_all_layers(model.module)
            else:
                unfreeze_all_layers(model)
        
        # Train
        train_loss, train_metrics, loss_components = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, writer, rank, scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clip_val=args.gradient_clip_val,
            aux_weight=args.aux_weight
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer, rank, use_tta=args.use_tta, aux_weight=args.aux_weight
        )
        
        # Update scheduler (except OneCycleLR which updates per batch)
        if args.scheduler == 'reduce':
            scheduler.step(val_metrics['iou'])
        elif args.scheduler == 'cosine':
            scheduler.step()
        # OneCycleLR already updated per batch in train_epoch
        
        # Logging
        if rank == 0:
            # Console output
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, "
                       f"Dice: {train_metrics['dice']:.4f}, F1: {train_metrics['f1']:.4f}")
            logger.info(f"Val - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
                       f"Dice: {val_metrics['dice']:.4f}, F1: {val_metrics['f1']:.4f}")
            # Build loss components string dynamically
            loss_comp_parts = []
            for key, value in loss_components.items():
                if isinstance(value, (int, float)):
                    loss_comp_parts.append(f"{key.capitalize()}: {value:.4f}")
            loss_comp_str = "Loss Components - " + ", ".join(loss_comp_parts)
            logger.info(loss_comp_str)
            
            # Tensorboard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            for metric, value in train_metrics.items():
                writer.add_scalar(f'Metrics/Train/{metric}', value, epoch)
            for metric, value in val_metrics.items():
                writer.add_scalar(f'Metrics/Val/{metric}', value, epoch)
            for component, value in loss_components.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'LossComponents/{component}', value, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_iou': best_iou,
                    'args': args
                }, output_dir / 'best_model.pth')
                logger.info(f"Saved best model with IoU: {best_iou:.4f}")
                
            # Early stopping check
            if early_stopping(val_metrics['iou'], epoch):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                logger.info(f"Best model was from epoch {early_stopping.best_epoch+1} with IoU: {best_iou:.4f}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_iou': best_iou,
                    'args': args
                }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Cleanup
    if rank == 0:
        writer.close()
    if world_size > 1:
        cleanup()

# ===========================
# Main Entry Point
# ===========================
def main():
    parser = argparse.ArgumentParser(description='Train segmentation model on spheroid dataset')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to training_big dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    
    # Model
    parser.add_argument('--model', type=str, default='unet',
                       choices=['hrnet', 'pspnet', 'unet', 'resunet_cbam', 'lightm_unet',
                               'resunet_ma', 'resunet_ma_mini', 'resunet_lc'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained backbone')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--img_size', type=int, default=1024,
                       help='Image size for training')
    
    # Loss weights
    parser.add_argument('--focal_weight', type=float, default=1.0,
                       help='Weight for focal loss')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                       help='Weight for dice loss')
    parser.add_argument('--iou_weight', type=float, default=0.5,
                       help='Weight for IoU loss')
    parser.add_argument('--boundary_weight', type=float, default=0.0,
                       help='Weight for boundary loss (0 to disable)')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                       help='Weight for auxiliary loss in PSPNet (0.4 is standard)')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['reduce', 'cosine', 'onecycle'],
                       help='Learning rate scheduler')
    
    # Others
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use Test Time Augmentation during validation')
    parser.add_argument('--use_instance_norm', action='store_true', default=True,
                       help='Use Instance Normalization instead of Batch Normalization')
    parser.add_argument('--find_lr', action='store_true',
                       help='Run learning rate finder before training')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='Minimum change in validation metric for early stopping')
    parser.add_argument('--use_cache', action='store_true',
                       help='Cache dataset for faster loading (creates .cache directory)')
    
    # Finetuning arguments
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model checkpoint for finetuning')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                       help='Number of epochs to freeze backbone layers during finetuning')
    parser.add_argument('--use_checkpoint', action='store_true',
                       help='Use gradient checkpointing to reduce memory usage (slower training)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value (0 to disable)')
    
    args = parser.parse_args()
    
    # Set up multi-GPU training
    world_size = args.gpus
    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(0, 1, args)

if __name__ == '__main__':
    main()
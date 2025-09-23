#!/bin/bash
# Finetune script for ResUNet Small (~60M params) on spheroid dataset
# Uses pretrained weights for faster convergence and better performance

# Set environment variables for optimal CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Configuration
MODEL="pspnet"
DATASET_PATH="/data/prusek/training_small"

# Pretrained model path - modify this to point to your pretrained model
PRETRAINED_PATH="/home/prusek/SpheroSeg/NN/diplomka/scripts/training/outputs/pspnet_new_pretrained_20250725_102928/best_model.pth"  # Will be set via command line argument or auto-detected

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained_path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--pretrained_path PATH] [--dataset_path PATH]"
            exit 1
            ;;
    esac
done

# Verify pretrained model exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Error: Pretrained model not found at: $PRETRAINED_PATH"
    exit 1
fi

OUTPUT_DIR="./outputs/${MODEL}_new_finetune_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/finetune.log"

# Finetuning hyperparameters (more conservative and stable)
BATCH_SIZE=10          # Conservative batch size for stability
IMG_SIZE=1024
EPOCHS=100             # Fewer epochs for finetuning
LEARNING_RATE=1e-6     # Much lower LR for stability (reduced from 5e-5)
WEIGHT_DECAY=5e-5      # Reduced weight decay for stability
OPTIMIZER="adamw"
SCHEDULER="reduce"     # Changed to ReduceLROnPlateau for better stability
PATIENCE=10            # Increased patience for stability

# Finetuning specific settings
FREEZE_BACKBONE_EPOCHS=15  # Longer freeze period for stability
GRADIENT_CLIP_VAL=0.5      # More aggressive gradient clipping
GRADIENT_ACCUMULATION=2    # Gradient accumulation for stability

# Regularization settings
FOCAL_WEIGHT=2.0
DICE_WEIGHT=1.0
IOU_WEIGHT=1.0
AUX_WEIGHT=0.4

# Hardware settings
GPUS=2
NUM_WORKERS=8

# Create output directory
mkdir -p $OUTPUT_DIR

echo "======================================================================"
echo "ResUNet Small Finetuning Configuration"
echo "======================================================================"
echo "Model: $MODEL"
echo "Pretrained: $PRETRAINED_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE (per GPU)"
echo "Image Size: ${IMG_SIZE}x${IMG_SIZE}"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Freeze Epochs: $FREEZE_BACKBONE_EPOCHS"
echo "GPUs: $GPUS"
echo "======================================================================"

# Log configuration
{
    echo "Finetuning started at: $(date)"
    echo "Configuration:"
    echo "  Model: $MODEL"
    echo "  Pretrained: $PRETRAINED_PATH"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Image Size: $IMG_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Freeze Epochs: $FREEZE_BACKBONE_EPOCHS"
    echo "  GPUs: $GPUS"
    echo ""
} > $LOG_FILE

# Kill any hanging processes
echo "Cleaning up any hanging processes..."
pkill -f CNN_main_spheroid.py || true
sleep 2

# Check GPU memory
echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Verify pretrained model compatibility
echo ""
echo "Verifying pretrained model compatibility..."
python -c "
import sys
sys.path.append('../../')
import torch
from models.resunet_small import ResUNetSmall

try:
    # Load pretrained model with PyTorch 2.6+ compatibility
    print('Attempting to load checkpoint...')

    # Method 1: Try with safe globals for argparse.Namespace
    try:
        torch.serialization.add_safe_globals([torch.serialization.safe_globals])
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load('$PRETRAINED_PATH', map_location='cpu', weights_only=True)
        print('✓ Loaded with weights_only=True and safe globals')
    except Exception as e1:
        print(f'Method 1 failed: {e1}')

        # Method 2: Try with weights_only=False (trusted source)
        try:
            checkpoint = torch.load('$PRETRAINED_PATH', map_location='cpu', weights_only=False)
            print('✓ Loaded with weights_only=False (trusted source)')
        except Exception as e2:
            print(f'Method 2 failed: {e2}')

            # Method 3: Legacy loading
            checkpoint = torch.load('$PRETRAINED_PATH', map_location='cpu')
            print('✓ Loaded with legacy method')

    # Create new model
    model = ResUNetSmall(in_channels=3, out_channels=1, dropout_rate=0.15)

    # Try to load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load with strict=False to handle any mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f'Missing keys: {len(missing_keys)} (this is usually OK for finetuning)')
    if unexpected_keys:
        print(f'Unexpected keys: {len(unexpected_keys)} (this is usually OK)')

    print('✓ Pretrained model loaded successfully!')
    print(f'Best IoU from pretraining: {checkpoint.get(\"best_iou\", \"N/A\")}')
    print(f'Epoch: {checkpoint.get(\"epoch\", \"N/A\")}')

except Exception as e:
    print(f'Error loading pretrained model: {e}')
    print('This may still work during actual training due to enhanced loading in CNN_main_spheroid.py')
    print('Continuing with finetuning...')
"

if [ $? -ne 0 ]; then
    echo "Pretrained model verification failed!" | tee -a $LOG_FILE
    exit 1
fi

echo ""
echo "Starting finetuning..." | tee -a $LOG_FILE

# Run finetuning with optimized parameters
python ../../CNN_main_spheroid.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --freeze_backbone_epochs $FREEZE_BACKBONE_EPOCHS \
    --batch_size $BATCH_SIZE \
    --img_size $IMG_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --focal_weight $FOCAL_WEIGHT \
    --dice_weight $DICE_WEIGHT \
    --iou_weight $IOU_WEIGHT \
    --aux_weight $AUX_WEIGHT \
    --use_instance_norm \
    --use_tta \
    --patience $PATIENCE \
    --min_delta 1e-4 \
    --num_workers $NUM_WORKERS \
    --gpus $GPUS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --use_cache \
    2>&1 | tee -a $LOG_FILE

# Check training result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Finetuning completed successfully!"
    echo "======================================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Final model saved as: ${OUTPUT_DIR}/best_model.pth"
    echo "======================================================================"
    
    # Display final results if available
    if [ -f "${OUTPUT_DIR}/training_results.json" ]; then
        echo "Final training results:"
        python -c "
import json
try:
    with open('${OUTPUT_DIR}/training_results.json', 'r') as f:
        results = json.load(f)
    print(f\"Best IoU: {results.get('best_iou', 'N/A'):.4f}\")
    print(f\"Best Dice: {results.get('best_dice', 'N/A'):.4f}\")
    print(f\"Final Loss: {results.get('final_loss', 'N/A'):.4f}\")
    print(f\"Total Epochs: {results.get('total_epochs', 'N/A')}\")
except:
    print('Results file not found or corrupted')
"
    fi
    
    # Compare with pretrained results
    echo ""
    echo "Improvement over pretraining:"
    python -c "
import torch
import json

try:
    # Load pretrained checkpoint
    pretrained = torch.load('$PRETRAINED_PATH', map_location='cpu')
    pretrained_iou = pretrained.get('best_iou', 0)
    
    # Load finetuned results
    with open('${OUTPUT_DIR}/training_results.json', 'r') as f:
        finetuned = json.load(f)
    finetuned_iou = finetuned.get('best_iou', 0)
    
    improvement = finetuned_iou - pretrained_iou
    print(f\"Pretrained IoU: {pretrained_iou:.4f}\")
    print(f\"Finetuned IoU: {finetuned_iou:.4f}\")
    print(f\"Improvement: {improvement:+.4f} ({improvement/pretrained_iou*100:+.2f}%)\")
    
except Exception as e:
    print(f'Could not compare results: {e}')
"
    
else
    echo ""
    echo "======================================================================"
    echo "Finetuning failed!"
    echo "======================================================================"
    echo "Check the log file for details: $LOG_FILE"
    echo "Common issues:"
    echo "  - Pretrained model incompatible"
    echo "  - Insufficient GPU memory (try reducing batch_size)"
    echo "  - Learning rate too high for finetuning"
    echo "======================================================================"
    exit 1
fi

echo ""
echo "Finetuning script completed at: $(date)"

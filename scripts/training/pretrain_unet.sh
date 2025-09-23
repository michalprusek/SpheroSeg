#!/bin/bash
# Pretrain script for basic UNet on spheroid dataset
# Optimized for 1024x1024 resolution

# Set environment variables for optimal CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Configuration
MODEL="unet"
DATASET_PATH="/data/prusek/training_big"
OUTPUT_DIR="./outputs/${MODEL}_pretrain_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/pretrain.log"

# Training hyperparameters optimized for pretraining
BATCH_SIZE=10          # Basic UNet can handle larger batch size
IMG_SIZE=1024
EPOCHS=150             # More epochs for pretraining
LEARNING_RATE=2e-4     # Higher LR for pretraining from scratch
WEIGHT_DECAY=1e-4
OPTIMIZER="adamw"
SCHEDULER="cosine"
PATIENCE=25            # More patience for pretraining

# Regularization settings
FOCAL_WEIGHT=1.0
DICE_WEIGHT=1.0
IOU_WEIGHT=0.5
BOUNDARY_WEIGHT=0.1

# Hardware settings
GPUS=2
NUM_WORKERS=8

# Create output directory
mkdir -p $OUTPUT_DIR

echo "======================================================================"
echo "UNet Pretraining Configuration"
echo "======================================================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE (per GPU)"
echo "Image Size: ${IMG_SIZE}x${IMG_SIZE}"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "GPUs: $GPUS"
echo "======================================================================"

# Log configuration
{
    echo "Pretraining started at: $(date)"
    echo "Configuration:"
    echo "  Model: $MODEL"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Image Size: $IMG_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Weight Decay: $WEIGHT_DECAY"
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

# Test model loading first
echo ""
echo "Testing model loading..."
python -c "
import sys
sys.path.append('../../')
from models.unet import UNet
import torch

model = UNet(in_channels=3, out_channels=1, dropout_rate=0.1)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
print(f'Model size: {total_params * 4 / 1024 / 1024:.1f} MB')

# Test forward pass
x = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
    print('Model test passed!')
"

if [ $? -ne 0 ]; then
    echo "Model test failed!" | tee -a $LOG_FILE
    exit 1
fi

echo ""
echo "Starting pretraining..." | tee -a $LOG_FILE

# Run pretraining with optimized parameters
python ../../CNN_main_spheroid.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
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
    --boundary_weight $BOUNDARY_WEIGHT \
    --use_instance_norm \
    --use_tta \
    --patience $PATIENCE \
    --min_delta 1e-4 \
    --num_workers $NUM_WORKERS \
    --gpus $GPUS \
    --gradient_accumulation_steps 1 \
    --gradient_clip_val 1.0 \
    --use_cache \
    2>&1 | tee -a $LOG_FILE

# Check training result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Pretraining completed successfully!"
    echo "======================================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Best model should be saved as: ${OUTPUT_DIR}/best_model.pth"
    echo "Use this model for finetuning with finetune_unet.sh"
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
except:
    print('Results file not found or corrupted')
"
    fi
else
    echo ""
    echo "======================================================================"
    echo "Pretraining failed!"
    echo "======================================================================"
    echo "Check the log file for details: $LOG_FILE"
    echo "Common issues:"
    echo "  - Insufficient GPU memory (try reducing batch_size)"
    echo "  - Dataset path incorrect"
    echo "  - CUDA out of memory (try --use_checkpoint flag)"
    echo "======================================================================"
    exit 1
fi

echo ""
echo "Pretraining script completed at: $(date)"
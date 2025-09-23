#!/bin/bash

# Pretrain Advanced ResUNet on training_big dataset
# Features: SimAM + Triplet Attention + Lightweight Self-Attention

echo "=========================================="
echo "Pretraining Advanced ResUNet with Modern Attention"
echo "=========================================="

# Set environment
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Go to project root directory
cd ../..
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training parameters
DATASET_PATH="/data/prusek/training_big"
BATCH_SIZE=1  # Minimal batch size due to memory constraints
GRADIENT_ACCUMULATION=8  # Effective batch size = 16
EPOCHS=100
PATIENCE=5
MIN_DELTA=1e-4

# Model specific parameters
MODEL_NAME="resunet_advanced"  # Using full advanced model with 78M params
LEARNING_RATE=5e-4  # Between ResUNet (1e-3) and HRNet (5e-4)
OPTIMIZER="adamw"
GRADIENT_CLIP=1.0  # Gradient clipping to prevent exploding gradients

# Create output directory
OUTPUT_DIR="scripts/training/outputs/${MODEL_NAME}_pretrained"
mkdir -p $OUTPUT_DIR

# Log file
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "Configuration:" | tee $LOG_FILE
echo "- Model: Advanced ResUNet (Pretraining)" | tee -a $LOG_FILE
echo "- Dataset: $DATASET_PATH" | tee -a $LOG_FILE
echo "- Batch Size: $BATCH_SIZE (Effective: $((BATCH_SIZE * GRADIENT_ACCUMULATION)) with gradient accumulation)" | tee -a $LOG_FILE
echo "- Gradient Accumulation Steps: $GRADIENT_ACCUMULATION" | tee -a $LOG_FILE
echo "- Gradient Clipping: $GRADIENT_CLIP" | tee -a $LOG_FILE
echo "- Learning Rate: $LEARNING_RATE" | tee -a $LOG_FILE
echo "- Optimizer: $OPTIMIZER" | tee -a $LOG_FILE
echo "- Output Directory: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# First, check if model loads correctly and has expected capacity
echo "Checking model capacity..." | tee -a $LOG_FILE
python -c "
import sys
sys.path.append('.')
from models.resunet_advanced import AdvancedResUNet
import torch

model = AdvancedResUNet(in_channels=3, out_channels=1, use_instance_norm=True)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')
print(f'Model size: {total_params * 4 / 1024 / 1024:.2f} MB')

# Test forward pass with actual training size
x = torch.randn(1, 3, 1024, 1024)
try:
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
    print('Model loaded successfully!')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" | tee -a $LOG_FILE

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Model check failed!" | tee -a $LOG_FILE
    exit 1
fi

echo "" | tee -a $LOG_FILE
echo "Starting pretraining..." | tee -a $LOG_FILE

# Run training with recommended parameters
python CNN_main_spheroid.py \
    --dataset_path $DATASET_PATH \
    --model $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --gradient_clip_val $GRADIENT_CLIP \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --patience $PATIENCE \
    --min_delta $MIN_DELTA \
    --use_instance_norm \
    --gpus 2 \
    --output_dir $OUTPUT_DIR \
    --boundary_weight 0.1 \
    2>&1 | tee -a $LOG_FILE

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "Pretraining completed successfully!" | tee -a $LOG_FILE
    echo "Model saved in: $OUTPUT_DIR" | tee -a $LOG_FILE
    
    # Run evaluation on test set
    echo "" | tee -a $LOG_FILE
    echo "Running evaluation on test set..." | tee -a $LOG_FILE
    
    python scripts/evaluation/evaluate_model.py \
        --model_path "${OUTPUT_DIR}/best_model.pth" \
        --dataset_path $DATASET_PATH \
        --model_name resunet_advanced \
        --use_tta \
        --output_dir $OUTPUT_DIR \
        2>&1 | tee -a $LOG_FILE
else
    echo "" | tee -a $LOG_FILE
    echo "Pretraining failed!" | tee -a $LOG_FILE
    exit 1
fi

echo "" | tee -a $LOG_FILE
echo "=========================================="
echo "Advanced ResUNet Pretraining Complete"
echo "=========================================="
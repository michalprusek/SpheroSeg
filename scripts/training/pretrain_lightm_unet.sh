#!/bin/bash
# Pretrain script for LightM-UNet on SpheroMix dataset
# LightM-UNet: ~1M parameters lightweight architecture

# Set environment variables for optimal CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Configuration
MODEL="lightm_unet"
DATASET_PATH="/data/prusek/training_big"
OUTPUT_DIR="./outputs/${MODEL}_pretrain_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/pretrain.log"

# Training hyperparameters optimized for LightM-UNet pretraining
BATCH_SIZE=1          # Lightweight model can handle reasonable batch size
IMG_SIZE=1024         # Use 1024x1024 for consistency with other models
EPOCHS=150            # More epochs for pretraining
LEARNING_RATE=1e-4    # Higher LR for pretraining from scratch
WEIGHT_DECAY=1e-5
OPTIMIZER="adamw"
SCHEDULER="cosine"
PATIENCE=25           # More patience for pretraining

# Regularization settings
FOCAL_WEIGHT=1.0
DICE_WEIGHT=1.0
IOU_WEIGHT=0.5
BOUNDARY_WEIGHT=0.0   # No boundary loss for pretraining

# Hardware settings
GPUS=2
NUM_WORKERS=8

# Create output directory
mkdir -p $OUTPUT_DIR

# Print configuration
echo "=== LightM-UNet Pretraining Configuration ===" | tee $LOG_FILE
echo "Model: $MODEL" | tee -a $LOG_FILE
echo "Dataset: $DATASET_PATH" | tee -a $LOG_FILE
echo "Output: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "Batch size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "Image size: $IMG_SIZE" | tee -a $LOG_FILE
echo "Epochs: $EPOCHS" | tee -a $LOG_FILE
echo "Learning rate: $LEARNING_RATE" | tee -a $LOG_FILE
echo "Weight decay: $WEIGHT_DECAY" | tee -a $LOG_FILE
echo "Optimizer: $OPTIMIZER" | tee -a $LOG_FILE
echo "Scheduler: $SCHEDULER" | tee -a $LOG_FILE
echo "GPUs: $GPUS" | tee -a $LOG_FILE
echo "Workers: $NUM_WORKERS" | tee -a $LOG_FILE
echo "Patience: $PATIENCE" | tee -a $LOG_FILE
echo "=============================================" | tee -a $LOG_FILE

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
    --find_lr \
    --patience $PATIENCE \
    --min_delta 1e-4 \
    --num_workers $NUM_WORKERS \
    --gpus $GPUS \
    --gradient_accumulation_steps 8 \
    --gradient_clip_val 1.0 \
    --use_cache \
    2>&1 | tee -a $LOG_FILE

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Pretraining Completed Successfully ===" | tee -a $LOG_FILE
    echo "Model saved in: $OUTPUT_DIR" | tee -a $LOG_FILE
    echo "Best model checkpoint: $OUTPUT_DIR/best_model.pth" | tee -a $LOG_FILE
    echo "Training log: $LOG_FILE" | tee -a $LOG_FILE
    echo "=========================================" | tee -a $LOG_FILE
else
    echo ""
    echo "=== Pretraining Failed ===" | tee -a $LOG_FILE
    echo "Check the log file for details: $LOG_FILE" | tee -a $LOG_FILE
    echo "=========================" | tee -a $LOG_FILE
    exit 1
fi
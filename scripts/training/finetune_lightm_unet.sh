#!/bin/bash
# Finetune script for LightM-UNet on high quality dataset
# LightM-UNet: ~1M parameters lightweight architecture

# Set environment variables for optimal CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

# Configuration
MODEL="lightm_unet"
DATASET_PATH="/data/prusek/training_small"
OUTPUT_DIR="./outputs/${MODEL}_finetune_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/finetune.log"

# Pretrained model path - automatically find the latest pretrained model
PRETRAINED_DIR="./outputs"
PRETRAINED_PATH=$(find $PRETRAINED_DIR -name "${MODEL}_pretrain_*" -type d | sort | tail -1)
if [ -n "$PRETRAINED_PATH" ]; then
    PRETRAINED_PATH="${PRETRAINED_PATH}/best_model.pth"
else
    echo "Error: No pretrained ${MODEL} model found in $PRETRAINED_DIR"
    echo "Please run pretrain_${MODEL}.sh first"
    exit 1
fi

# Training hyperparameters optimized for LightM-UNet finetuning
FREEZE_BACKBONE_EPOCHS=10  # Freeze backbone for initial epochs
BATCH_SIZE=1              # Lightweight model can handle reasonable batch size
IMG_SIZE=1024             # Use 1024x1024 for consistency
EPOCHS=75                 # Fewer epochs for finetuning
LEARNING_RATE=1e-5        # Lower LR for finetuning
WEIGHT_DECAY=1e-5
OPTIMIZER="adamw"
SCHEDULER="cosine"
PATIENCE=5               # Less patience for finetuning

# Regularization settings
FOCAL_WEIGHT=1.0
DICE_WEIGHT=1.0
IOU_WEIGHT=0.5
BOUNDARY_WEIGHT=0.0       # Add boundary loss for finetuning

# Hardware settings
GPUS=2
NUM_WORKERS=8
GRADIENT_ACCUMULATION=8
GRADIENT_CLIP_VAL=1.0

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Error: Pretrained model not found at $PRETRAINED_PATH"
    echo "Available pretrained models:"
    find $PRETRAINED_DIR -name "${MODEL}_pretrain_*" -type d 2>/dev/null || echo "None found"
    exit 1
fi

# Print configuration
echo "=== LightM-UNet Finetuning Configuration ===" | tee $LOG_FILE
echo "Model: $MODEL" | tee -a $LOG_FILE
echo "Dataset: $DATASET_PATH" | tee -a $LOG_FILE
echo "Pretrained model: $PRETRAINED_PATH" | tee -a $LOG_FILE
echo "Output: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "Freeze backbone epochs: $FREEZE_BACKBONE_EPOCHS" | tee -a $LOG_FILE
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
    --boundary_weight $BOUNDARY_WEIGHT \
    --use_instance_norm \
    --use_tta \
    --patience $PATIENCE \
    --min_delta 5e-5 \
    --num_workers $NUM_WORKERS \
    --gpus $GPUS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --use_cache \
    2>&1 | tee -a $LOG_FILE

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Finetuning Completed Successfully ===" | tee -a $LOG_FILE
    echo "Model saved in: $OUTPUT_DIR" | tee -a $LOG_FILE
    echo "Best model checkpoint: $OUTPUT_DIR/best_model.pth" | tee -a $LOG_FILE
    echo "Training log: $LOG_FILE" | tee -a $LOG_FILE
    echo "=======================================" | tee -a $LOG_FILE
else
    echo ""
    echo "=== Finetuning Failed ===" | tee -a $LOG_FILE
    echo "Check the log file for details: $LOG_FILE" | tee -a $LOG_FILE
    echo "========================" | tee -a $LOG_FILE
    exit 1
fi
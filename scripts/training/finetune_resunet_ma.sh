#!/bin/bash

# Finetune Advanced ResUNet on training_small dataset
# Features: SimAM + Triplet Attention + Lightweight Self-Attention

echo "=========================================="
echo "Finetuning Advanced ResUNet with Modern Attention"
echo "=========================================="

# Set environment
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Go to project root directory
cd ../..
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training parameters
DATASET_PATH="/data/prusek/training_small"
BATCH_SIZE=1  # Minimal batch size due to memory constraints
GRADIENT_ACCUMULATION=8  # Effective batch size = 16
EPOCHS=100
PATIENCE=5
MIN_DELTA=1e-4

# Model specific parameters
MODEL_NAME="resunet_advanced"
LEARNING_RATE=1e-5  # Lower LR for finetuning
OPTIMIZER="adamw"

# Pretrained model path
PRETRAINED_PATH="scripts/training/outputs/resunet_advanced_pretrained/best_model.pth"

# Create output directory
OUTPUT_DIR="scripts/training/outputs/${MODEL_NAME}_finetuned"
mkdir -p $OUTPUT_DIR

# Log file
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "Configuration:" | tee $LOG_FILE
echo "- Model: Advanced ResUNet (Finetuning)" | tee -a $LOG_FILE
echo "- Dataset: $DATASET_PATH" | tee -a $LOG_FILE
echo "- Pretrained Model: $PRETRAINED_PATH" | tee -a $LOG_FILE
echo "- Batch Size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "- Learning Rate: $LEARNING_RATE" | tee -a $LOG_FILE
echo "- Optimizer: $OPTIMIZER" | tee -a $LOG_FILE
echo "- Output Directory: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Pretrained model not found at: $PRETRAINED_PATH" | tee -a $LOG_FILE
    echo "Please run pretraining first!" | tee -a $LOG_FILE
    exit 1
fi

echo "" | tee -a $LOG_FILE
echo "Starting finetuning..." | tee -a $LOG_FILE

# Run finetuning with lower learning rate
python CNN_main_spheroid.py \
    --dataset_path $DATASET_PATH \
    --model resunet_advanced \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --patience $PATIENCE \
    --min_delta $MIN_DELTA \
    --use_instance_norm \
    --gpus 2 \
    --output_dir $OUTPUT_DIR \
    --boundary_weight 0.0 \
    --pretrained_path $PRETRAINED_PATH \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --freeze_backbone_epochs 10 \
    2>&1 | tee -a $LOG_FILE

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "Finetuning completed successfully!" | tee -a $LOG_FILE
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
    echo "Finetuning failed!" | tee -a $LOG_FILE
    exit 1
fi

echo "" | tee -a $LOG_FILE
echo "=========================================="
echo "Advanced ResUNet Finetuning Complete"
echo "=========================================="
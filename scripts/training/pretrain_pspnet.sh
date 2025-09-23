#!/bin/bash

# PSPNet New Architecture Pretraining Script
# Optimized for 2x L40S GPUs with 70.29M parameter model
# Includes auxiliary loss handling and memory optimization

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check if we're already in a tmux session
if [ -z "$TMUX" ]; then
    # Create new tmux session
    SESSION_NAME="pspnet_new_pretrain_$(date +%Y%m%d_%H%M%S)"
    echo "============================================================"
    echo "PSPNet New Architecture Pretraining"
    echo "============================================================"
    echo "Creating tmux session: $SESSION_NAME"
    echo "To detach: Ctrl+B then D"
    echo "To reattach: tmux attach -t $SESSION_NAME"
    echo ""
    echo "Model: PSPNet with ResNet101 backbone (70.29M parameters)"
    echo "Dataset: Large spheroid dataset (~19,490 training images)"
    echo "GPUs: 2x NVIDIA L40S (44.4 GB each)"
    echo "Batch size: 6 (3 per GPU) with mixed precision"
    echo ""
    sleep 3
    tmux new-session -d -s "$SESSION_NAME" "cd $SCRIPT_DIR && bash $0"
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "============================================================"
echo "PSPNet New Architecture Pretraining"
echo "Running in tmux session: $(tmux display-message -p '#S')"
echo "Start time: $(date)"
echo "============================================================"

# GPU status check
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# Create directories
mkdir -p ./logs
mkdir -p ./outputs

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/pspnet_new_pretrained_$TIMESTAMP"
LOG_FILE="./logs/pspnet_new_pretrain_$TIMESTAMP.log"

echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Training parameters optimized for new PSPNet
DATASET_PATH="/data/prusek/training_big"
MODEL_NAME="pspnet"
EPOCHS=25
BATCH_SIZE=6  # Total batch size across 2 GPUs (3 per GPU)
LEARNING_RATE=3e-6  # Lower LR for ResNet101 backbone
WEIGHT_DECAY=1e-4
IMG_SIZE=1024
OPTIMIZER="adamw"
SCHEDULER="cosine"
NUM_WORKERS=6
PATIENCE=15

# Loss weights for auxiliary supervision
FOCAL_WEIGHT=2.0
DICE_WEIGHT=1.0
IOU_WEIGHT=1.0
AUX_WEIGHT=0.4  # Standard auxiliary loss weight

echo "Training Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Model: $MODEL_NAME (new architecture)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE (${BATCH_SIZE}/2 per GPU)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo "  Optimizer: $OPTIMIZER"
echo "  Scheduler: $SCHEDULER"
echo "  Auxiliary loss weight: $AUX_WEIGHT"
echo ""

# Start training
echo "Starting PSPNet pretraining..."
echo "============================================================"

python3 $PROJECT_ROOT/CNN_main_spheroid.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --img_size $IMG_SIZE \
    --optimizer "$OPTIMIZER" \
    --scheduler "$SCHEDULER" \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE \
    --gpus 2 \
    --focal_weight $FOCAL_WEIGHT \
    --dice_weight $DICE_WEIGHT \
    --iou_weight $IOU_WEIGHT \
    --aux_weight $AUX_WEIGHT \
    --use_instance_norm \
    --gradient_clip_val 1.0 \
    --min_delta 1e-4 \
    --use_cache \
    --gradient_accumulation_steps 1 \
    2>&1 | tee "$LOG_FILE"

# Training completion
TRAINING_EXIT_CODE=$?
echo ""
echo "============================================================"
echo "PSPNet Pretraining Complete!"
echo "End time: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "============================================================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo "✓ Best model saved to: $OUTPUT_DIR/best_model.pth"
    echo "✓ Training log: $LOG_FILE"
    
    # Display final model info
    if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
        echo ""
        echo "Model Information:"
        python3 -c "
import torch
import sys
sys.path.append('$PROJECT_ROOT')

try:
    checkpoint = torch.load('$OUTPUT_DIR/best_model.pth', map_location='cpu', weights_only=False)
    print(f'  Final epoch: {checkpoint.get(\"epoch\", \"N/A\")}')
    print(f'  Best IoU: {checkpoint.get(\"best_iou\", \"N/A\"):.4f}')
    print(f'  Best loss: {checkpoint.get(\"best_loss\", \"N/A\"):.6f}')
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f'  Learning rate: {getattr(args, \"lr\", \"N/A\")}')
        print(f'  Batch size: {getattr(args, \"batch_size\", \"N/A\")}')
except Exception as e:
    print(f'  Error reading checkpoint: {e}')
"
    fi
else
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "✗ Check log file for details: $LOG_FILE"
fi

echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "============================================================"
echo "Next Steps:"
echo "1. Check training results in: $OUTPUT_DIR"
echo "2. Review training log: $LOG_FILE"
echo "3. If successful, proceed with finetuning using:"
echo "   ./finetune_pspnet_new.sh $OUTPUT_DIR/best_model.pth"
echo "============================================================"

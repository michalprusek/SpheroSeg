#!/bin/bash
# Pretrain HRNet on large spheroid dataset

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check if we're already in a tmux session
if [ -z "$TMUX" ]; then
    # Create new tmux session
    SESSION_NAME="hrnet_pretrain_$(date +%Y%m%d_%H%M%S)"
    echo "Creating tmux session: $SESSION_NAME"
    echo "To detach: Ctrl+B then D"
    echo "To reattach: tmux attach -t $SESSION_NAME"
    echo ""
    sleep 2
    tmux new-session -d -s "$SESSION_NAME" "cd $SCRIPT_DIR && bash $0"
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

echo "============================================================"
echo "HRNet Pretraining on Large Dataset"
echo "Running in tmux session: $(tmux display-message -p '#S')"
echo "============================================================"

# Create log directory
mkdir -p ./logs
LOG_FILE="./logs/hrnet_pretrain_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

python $PROJECT_ROOT/CNN_main_spheroid.py \
    --dataset_path /data/prusek/training_big \
    --output_dir ./outputs/hrnet_pretrained \
    --model hrnet \
    --epochs 20 \
    --batch_size 10 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --img_size 1024 \
    --optimizer adamw \
    --scheduler onecycle \
    --num_workers 4 \
    --patience 10 \
    --gpus 2 \
    --focal_weight 1.0 \
    --dice_weight 1.0 \
    --iou_weight 0.5 \
    --use_instance_norm \
    --min_delta 5e-4 \
    --use_cache 2>&1 | tee "$LOG_FILE"

echo "Pretraining complete! Best model saved to outputs/hrnet_pretrained/best_model.pth"
#!/bin/bash
# Finetune HRNet on small high-quality dataset

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check if we're already in a tmux session
if [ -z "$TMUX" ]; then
    # Create new tmux session
    SESSION_NAME="hrnet_finetune_$(date +%Y%m%d_%H%M%S)"
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
echo "HRNet Finetuning on High-Quality Dataset"
echo "Running in tmux session: $(tmux display-message -p '#S')"
echo "============================================================"

# Create log directory
mkdir -p ./logs
LOG_FILE="./logs/hrnet_finetune_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

python $PROJECT_ROOT/CNN_main_spheroid.py \
    --dataset_path /data/prusek/training_small \
    --output_dir ./outputs/hrnet_finetuned \
    --model hrnet \
    --pretrained_path ./outputs/hrnet_pretrained/best_model.pth \
    --freeze_backbone_epochs 4 \
    --epochs 30 \
    --batch_size 10 \
    --lr 5e-6 \
    --weight_decay 5e-5 \
    --img_size 1024 \
    --optimizer adamw \
    --scheduler cosine \
    --num_workers 4 \
    --patience 15 \
    --gpus 2 \
    --focal_weight 1.0 \
    --dice_weight 1.5 \
    --iou_weight 0.5 \
    --use_instance_norm \
    --min_delta 1e-4 \
    --use_cache 2>&1 | tee "$LOG_FILE"

echo "Finetuning complete! Best model saved to outputs/hrnet_finetuned/best_model.pth"
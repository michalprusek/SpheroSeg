#!/bin/bash
# Eval all A3 trained models on HTS-Seg held-out OOD test (89 full images).
# Output JSON has "unified" populated (all rows, no HQ/DTS stratification).
set -e
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
SCRIPT=/disk1/prusek/SpheroSeg/code/scripts/a3/evaluate_a3.py
OUT=/disk1/prusek/SpheroSeg/eval_a3/results_htsseg
LOG=/disk1/prusek/SpheroSeg/logs/batch_eval_htsseg.log
CKPT=/disk1/prusek/SpheroSeg/checkpoints
DATASET=/disk1/prusek/SpheroSeg/data/HTS_Seg_eval

JOBS=(
  "hrnet|$CKPT/hrnet_a3_pretrained_seed42/best_model.pth|hrnet_a3_pretrained.json"
  "hrnet|$CKPT/hrnet_a3_finetuned_seed42_a5000/best_model.pth|hrnet_a3_finetuned_a5000.json"
  "unet|$CKPT/unet_a3_pretrained_seed42_a5000/best_model.pth|unet_a3_pretrained_a5000.json"
  "unet|$CKPT/unet_a3_finetuned_seed42_a5000/best_model.pth|unet_a3_finetuned_a5000.json"
  "pspnet|$CKPT/pspnet_a3_pretrained_seed42/best_model.pth|pspnet_a3_pretrained.json"
  "pspnet|$CKPT/pspnet_a3_finetuned_seed42/best_model.pth|pspnet_a3_finetuned.json"
  "resunet_cbam|$CKPT/resunet_cbam_a3_pretrained_seed42/best_model.pth|resunet_cbam_a3_pretrained.json"
)

mkdir -p "$OUT"
echo "===== HTS-Seg batch eval start $(date -Iseconds) =====" > "$LOG"

for job in "${JOBS[@]}"; do
  IFS='|' read -r model weights out <<< "$job"
  if [ -f "$OUT/$out" ]; then
    echo "[skip] $out already exists" >> "$LOG"
    continue
  fi
  echo "----- $(date -Iseconds) eval $model on HTS-Seg -----" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" \
    --weights "$weights" --model "$model" \
    --dataset "$DATASET" \
    --output "$OUT/$out" \
    --batch_size 4 --num_workers 4 >> "$LOG" 2>&1
  echo "----- done rc=$? -----" >> "$LOG"
done

echo "===== HTS-Seg batch eval finished $(date -Iseconds) =====" >> "$LOG"

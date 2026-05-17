#!/bin/bash
# Eval round-2 retrained weights: lc + ma + ma_mini (A3 protocol, _a3_seed42 dirs).
set -e
GPU=${1:-0}
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
SCRIPT=/disk1/prusek/SpheroSeg/code/scripts/a3/evaluate_a3.py
OUT_SM=/disk1/prusek/SpheroSeg/eval_a3/results_full/spheromix
OUT_HTS=/disk1/prusek/SpheroSeg/eval_a3/results_full/htsseg
LOG=/disk1/prusek/SpheroSeg/logs/eval_round2_gpu${GPU}.log
CKPT=/disk1/prusek/SpheroSeg/checkpoints

mkdir -p "$OUT_SM" "$OUT_HTS"

JOBS=(
  "resunet_lc|$CKPT/resunet_lc_a3_pretrained_seed42/best_model.pth|resunet_lc_a3_pretrained.json"
  "resunet_lc|$CKPT/resunet_lc_a3_finetuned_seed42/best_model.pth|resunet_lc_a3_finetuned.json"
  "resunet_ma|$CKPT/resunet_ma_a3_pretrained_seed42/best_model.pth|resunet_ma_a3_pretrained.json"
  "resunet_ma|$CKPT/resunet_ma_a3_finetuned_seed42/best_model.pth|resunet_ma_a3_finetuned.json"
  "resunet_ma_mini|$CKPT/resunet_ma_mini_a3_pretrained_seed42_a5000/best_model.pth|resunet_ma_mini_a3_pretrained.json"
  "resunet_ma_mini|$CKPT/resunet_ma_mini_a3_finetuned_seed42_a5000/best_model.pth|resunet_ma_mini_a3_finetuned.json"
)

echo "===== eval round-2 start $(date -Iseconds) on GPU $GPU =====" > "$LOG"

for job in "${JOBS[@]}"; do
  IFS='|' read -r model weights out <<< "$job"
  if [ ! -f "$weights" ]; then
    echo "[skip-missing] $weights" >> "$LOG"; continue
  fi
  if [ -f "$OUT_SM/$out" ]; then
    echo "[skip-have] SM $out" >> "$LOG"
  else
    echo "----- $(date -Iseconds) $model on SpheroMix -----" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCRIPT" --weights "$weights" --model "$model" \
      --dataset /disk1/prusek/SpheroSeg/data/SpheroMix --output "$OUT_SM/$out" \
      --batch_size 4 --num_workers 4 >> "$LOG" 2>&1 || echo "[err-SM] $out rc=$?" >> "$LOG"
  fi
  if [ -f "$OUT_HTS/$out" ]; then
    echo "[skip-have] HTS $out" >> "$LOG"
  else
    echo "----- $(date -Iseconds) $model on HTS-Seg -----" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCRIPT" --weights "$weights" --model "$model" \
      --dataset /disk1/prusek/SpheroSeg/data/HTS_Seg_eval --output "$OUT_HTS/$out" \
      --batch_size 4 --num_workers 4 >> "$LOG" 2>&1 || echo "[err-HTS] $out rc=$?" >> "$LOG"
  fi
done

echo "===== eval round-2 done $(date -Iseconds) =====" >> "$LOG"

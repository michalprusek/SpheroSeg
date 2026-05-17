#!/bin/bash
# Full-roster eval. Args: <gpu_idx> <variant: a3|orig>
# - SpheroMix run gives HQ + DTS stratified via is_hq_test() in evaluate_a3.py.
# - HTS-Seg run gives unified-only IoU on OOD held-out.
set -e
GPU=${1:-1}
VARIANT=${2:-a3}
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
SCRIPT=/disk1/prusek/SpheroSeg/code/scripts/a3/evaluate_a3.py
OUT_SM=/disk1/prusek/SpheroSeg/eval_a3/results_full/spheromix
OUT_HTS=/disk1/prusek/SpheroSeg/eval_a3/results_full/htsseg
LOG=/disk1/prusek/SpheroSeg/logs/batch_eval_full_${VARIANT}_gpu${GPU}.log
CKPT=/disk1/prusek/SpheroSeg/checkpoints
WO=/disk1/prusek/SpheroSeg/weights_orig

mkdir -p "$OUT_SM" "$OUT_HTS"

if [ "$VARIANT" = "a3" ]; then
  JOBS=(
    "unet|$CKPT/unet_a3_pretrained_seed42_a5000/best_model.pth|unet_a3_pretrained.json"
    "unet|$CKPT/unet_a3_finetuned_seed42_a5000/best_model.pth|unet_a3_finetuned.json"
    "hrnet|$CKPT/hrnet_a3_pretrained_seed42/best_model.pth|hrnet_a3_pretrained.json"
    "hrnet|$CKPT/hrnet_a3_finetuned_seed42_a5000/best_model.pth|hrnet_a3_finetuned.json"
    "pspnet|$CKPT/pspnet_a3_pretrained_seed42/best_model.pth|pspnet_a3_pretrained.json"
    "pspnet|$CKPT/pspnet_a3_finetuned_seed42/best_model.pth|pspnet_a3_finetuned.json"
    "resunet_cbam|$CKPT/resunet_cbam_a3_pretrained_seed42/best_model.pth|resunet_cbam_a3_pretrained.json"
    "resunet_cbam|$CKPT/resunet_cbam_a3_finetuned_seed42_a5000/best_model.pth|resunet_cbam_a3_finetuned.json"
  )
elif [ "$VARIANT" = "orig" ]; then
  JOBS=(
    "resunet_lc|$WO/resunet_lc_pretrained.pth|resunet_lc_orig_pretrained.json"
    "resunet_lc|$WO/resunet_lc_finetuned.pth|resunet_lc_orig_finetuned.json"
    "resunet_ma|$WO/resunet_ma_pretrained.pth|resunet_ma_orig_pretrained.json"
    "resunet_ma|$WO/resunet_ma_finetuned.pth|resunet_ma_orig_finetuned.json"
    "resunet_ma_mini|$WO/resunet_ma_mini_pretrained.pth|resunet_ma_mini_orig_pretrained.json"
    "resunet_ma_mini|$WO/resunet_ma_mini_finetuned.pth|resunet_ma_mini_orig_finetuned.json"
    "lightm_unet|$WO/lightm_unet_pretrained.pth|lightm_unet_orig_pretrained.json"
    "lightm_unet|$WO/lightm_unet_finetuned.pth|lightm_unet_orig_finetuned.json"
  )
else
  echo "unknown variant: $VARIANT"; exit 1
fi

echo "===== full-roster eval $VARIANT start $(date -Iseconds) on GPU $GPU =====" > "$LOG"

for job in "${JOBS[@]}"; do
  IFS='|' read -r model weights out <<< "$job"
  if [ ! -f "$weights" ]; then
    echo "[skip-missing] $weights" >> "$LOG"; continue
  fi
  # SpheroMix
  if [ -f "$OUT_SM/$out" ]; then
    echo "[skip-have] SM $out" >> "$LOG"
  else
    echo "----- $(date -Iseconds) $model on SpheroMix -----" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCRIPT" --weights "$weights" --model "$model" \
      --dataset /disk1/prusek/SpheroSeg/data/SpheroMix --output "$OUT_SM/$out" \
      --batch_size 4 --num_workers 4 >> "$LOG" 2>&1 || echo "[err-SM] $out rc=$?" >> "$LOG"
  fi
  # HTS-Seg
  if [ -f "$OUT_HTS/$out" ]; then
    echo "[skip-have] HTS $out" >> "$LOG"
  else
    echo "----- $(date -Iseconds) $model on HTS-Seg -----" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCRIPT" --weights "$weights" --model "$model" \
      --dataset /disk1/prusek/SpheroSeg/data/HTS_Seg_eval --output "$OUT_HTS/$out" \
      --batch_size 4 --num_workers 4 >> "$LOG" 2>&1 || echo "[err-HTS] $out rc=$?" >> "$LOG"
  fi
done

echo "===== eval $VARIANT done $(date -Iseconds) =====" >> "$LOG"

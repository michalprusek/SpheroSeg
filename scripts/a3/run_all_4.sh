#!/usr/bin/env bash
# Run all 4 A3 retraining jobs sequentially on A100 (CUDA_VISIBLE_DEVICES=0).
# Order chosen by expected wall time (fastest first → fail fast on errors):
#   1. HRNet  (~10h, eff BS=16, BS=8, GA=2)
#   2. U-Net  (~10h, eff BS=16, BS=8, GA=2)
#   3. PSPNet (~18h, eff BS=16, BS=4, GA=4)
#   4. CBAM-ResUNet (~21h, eff BS=16, BS=4, GA=4)
# Total wall: ~59h ≈ 2.5 days (assuming no early-stop kicks in earlier)

set -u  # NOT -e: we want all 4 to run even if one fails

REPO=/disk1/prusek/SpheroSeg/code
LOG_ROOT=/disk1/prusek/SpheroSeg/logs
mkdir -p $LOG_ROOT

PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
cd $REPO

run_one() {
  local model=$1
  local log=$LOG_ROOT/${model}_a3.log
  echo "================================================================"
  echo "[run_all_4] $(date -Iseconds) — starting $model"
  echo "[run_all_4] log → $log"
  echo "================================================================"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/a3/run_a3_launcher.py --model $model --gpus 1 --num-workers 8 \
    > $log 2>&1
  rc=$?
  echo "[run_all_4] $(date -Iseconds) — $model finished with rc=$rc"
  return $rc
}

# Order: fastest first
for m in hrnet unet pspnet resunet_cbam; do
  run_one $m
done

echo "================================================================"
echo "[run_all_4] all done at $(date -Iseconds)"
ls -la /disk1/prusek/SpheroSeg/checkpoints/*/best_model.pth 2>/dev/null

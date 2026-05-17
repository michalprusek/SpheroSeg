#!/usr/bin/env bash
set -u
LOG=/disk1/prusek/SpheroSeg/logs/a100_recovery_chain.log
exec >>"$LOG" 2>&1
echo "================================================================"
echo "[$(date -Iseconds)] A100 recovery chain START (PID=$$)"
echo "================================================================"

PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
CODE=/disk1/prusek/SpheroSeg/code
cd "$CODE"

run_stage() {
  local model=$1
  local stage=$2
  local LOG_FILE
  if [ "$stage" = "finetune" ]; then
    LOG_FILE="/disk1/prusek/SpheroSeg/logs/${model}_a3_finetune.log"
  else
    LOG_FILE="/disk1/prusek/SpheroSeg/logs/${model}_a3.log"
  fi
  echo "----------------------------------------------------------------"
  echo "[$(date -Iseconds)] starting $model $stage  log=$LOG_FILE"
  echo "----------------------------------------------------------------"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/a3/run_a3_launcher.py --model "$model" --stage "$stage" \
      --gpus 1 --num-workers 8 \
    > "$LOG_FILE" 2>&1
  rc=$?
  echo "[$(date -Iseconds)] $model $stage rc=$rc"
}

run_stage pspnet         pretrain
run_stage resunet_cbam   pretrain
run_stage hrnet          finetune
run_stage pspnet         finetune
run_stage resunet_cbam   finetune

echo "================================================================"
echo "[$(date -Iseconds)] A100 recovery chain ALL DONE"
echo "================================================================"

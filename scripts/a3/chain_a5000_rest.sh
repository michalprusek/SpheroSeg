#!/bin/bash
# A5000 chain: resunet_ma_mini only (lightm_unet dropped from paper per user decision 2026-05-11)
# bs=8 ga=2 — resunet_ma_mini is small (~22M params), A5000 24GB should fit; EBS=16 maintained.
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
cd /disk1/prusek/SpheroSeg/code
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR=/disk1/prusek/SpheroSeg/logs
MASTER=$LOGDIR/a3_chain_a5000.log
echo "===== A5000 chain restart resunet_ma_mini bs=8 ga=2  $(date -Iseconds) =====" >> $MASTER

for STAGE in pretrain finetune; do
  LOG=$LOGDIR/resunet_ma_mini_a3_${STAGE}_round2.log
  echo "===== [resunet_ma_mini/$STAGE  bs=8 ga=2] start $(date -Iseconds) =====" > $LOG
  echo "    -> resunet_ma_mini $STAGE  bs=8 ga=2" >> $MASTER
  $PY scripts/a3/run_a3_launcher.py --model resunet_ma_mini --stage $STAGE \
    --gpus 1 --num-workers 4 --per-gpu-bs 2 --grad-accum 8 \
    --output-suffix a5000 >> $LOG 2>&1
  rc=$?
  echo "===== [resunet_ma_mini/$STAGE] done rc=$rc $(date -Iseconds) =====" >> $LOG
  echo "    <- resunet_ma_mini $STAGE rc=$rc" >> $MASTER
  [ $rc -ne 0 ] && { echo "    chain aborted" >> $MASTER; exit $rc; }
done
echo "===== A5000 chain complete $(date -Iseconds) =====" >> $MASTER

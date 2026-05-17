#!/bin/bash
# A100 chain RECOVERY after 19:41 reboot: warm-restart lc FT from round1 best (0.9095), then ma pretrain+finetune
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
cd /disk1/prusek/SpheroSeg/code
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR=/disk1/prusek/SpheroSeg/logs
MASTER=$LOGDIR/a3_chain_a100_recovery.log
echo "===== A100 chain RECOVERY $(date -Iseconds) =====" >> $MASTER

# Stage 1: lc finetune warm-restart from round1 FT best 0.9095
LCFT_WARM=/disk1/prusek/SpheroSeg/checkpoints/resunet_lc_a3_finetuned_seed42_round1/best_model.pth
LOG=$LOGDIR/resunet_lc_a3_finetune_round2.log
echo "===== [resunet_lc/finetune WARM] start $(date -Iseconds) =====" > $LOG
echo "    -> resunet_lc finetune warm from $LCFT_WARM" >> $MASTER
$PY scripts/a3/run_a3_launcher.py --model resunet_lc --stage finetune \
  --gpus 1 --num-workers 4 --per-gpu-bs 4 --grad-accum 4 \
  --pretrained-path "$LCFT_WARM" >> $LOG 2>&1
rc=$?
echo "===== [resunet_lc/finetune] done rc=$rc $(date -Iseconds) =====" >> $LOG
echo "    <- resunet_lc finetune rc=$rc" >> $MASTER
[ $rc -ne 0 ] && { echo "    chain aborted" >> $MASTER; exit $rc; }

# Stage 2 & 3: ma pretrain + finetune (clean, no warm restart needed)
for STAGE in pretrain finetune; do
  LOG=$LOGDIR/resunet_ma_a3_${STAGE}_round2.log
  echo "===== [resunet_ma/$STAGE] start $(date -Iseconds) =====" > $LOG
  echo "    -> resunet_ma $STAGE" >> $MASTER
  $PY scripts/a3/run_a3_launcher.py --model resunet_ma --stage $STAGE \
    --gpus 1 --num-workers 4 --per-gpu-bs 4 --grad-accum 4 >> $LOG 2>&1
  rc=$?
  echo "===== [resunet_ma/$STAGE] done rc=$rc $(date -Iseconds) =====" >> $LOG
  echo "    <- resunet_ma $STAGE rc=$rc" >> $MASTER
  [ $rc -ne 0 ] && { echo "    chain aborted" >> $MASTER; exit $rc; }
done
echo "===== A100 recovery chain complete $(date -Iseconds) =====" >> $MASTER

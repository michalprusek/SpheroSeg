#!/bin/bash
# A5000 chain RECOVERY after 19:41 reboot: warm-restart mini PT from round1 ep10 weights, then mini FT
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
cd /disk1/prusek/SpheroSeg/code
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR=/disk1/prusek/SpheroSeg/logs
MASTER=$LOGDIR/a3_chain_a5000_recovery.log
echo "===== A5000 chain RECOVERY $(date -Iseconds) =====" >> $MASTER

# Stage 1: mini pretrain warm-restart from round1 checkpoint_epoch_10.pth
MINI_PT_WARM=/disk1/prusek/SpheroSeg/checkpoints/resunet_ma_mini_a3_pretrained_seed42_a5000_round1/checkpoint_epoch_10.pth
LOG=$LOGDIR/resunet_ma_mini_a3_pretrain_round2.log
echo "===== [resunet_ma_mini/pretrain WARM] start $(date -Iseconds) =====" > $LOG
echo "    -> resunet_ma_mini pretrain warm from $MINI_PT_WARM" >> $MASTER
$PY scripts/a3/run_a3_launcher.py --model resunet_ma_mini --stage pretrain \
  --gpus 1 --num-workers 4 --per-gpu-bs 2 --grad-accum 8 \
  --output-suffix a5000 \
  --pretrained-path "$MINI_PT_WARM" >> $LOG 2>&1
rc=$?
echo "===== [resunet_ma_mini/pretrain] done rc=$rc $(date -Iseconds) =====" >> $LOG
echo "    <- resunet_ma_mini pretrain rc=$rc" >> $MASTER
[ $rc -ne 0 ] && { echo "    chain aborted" >> $MASTER; exit $rc; }

# Stage 2: mini finetune (clean, auto-discover pretrain best)
LOG=$LOGDIR/resunet_ma_mini_a3_finetune_round2.log
echo "===== [resunet_ma_mini/finetune] start $(date -Iseconds) =====" > $LOG
echo "    -> resunet_ma_mini finetune" >> $MASTER
$PY scripts/a3/run_a3_launcher.py --model resunet_ma_mini --stage finetune \
  --gpus 1 --num-workers 4 --per-gpu-bs 2 --grad-accum 8 \
  --output-suffix a5000 >> $LOG 2>&1
rc=$?
echo "===== [resunet_ma_mini/finetune] done rc=$rc $(date -Iseconds) =====" >> $LOG
echo "    <- resunet_ma_mini finetune rc=$rc" >> $MASTER
[ $rc -ne 0 ] && { echo "    chain aborted" >> $MASTER; exit $rc; }
echo "===== A5000 recovery chain complete $(date -Iseconds) =====" >> $MASTER

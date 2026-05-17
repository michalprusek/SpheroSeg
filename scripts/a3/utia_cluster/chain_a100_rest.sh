#!/bin/bash
# A100 chain: resunet_lc + resunet_ma, both bs=4 ga=4 (EBS=16 default, A100 80GB has plenty headroom)
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
cd /disk1/prusek/SpheroSeg/code
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR=/disk1/prusek/SpheroSeg/logs
MASTER=$LOGDIR/a3_chain_a100.log
echo "===== A100 chain restart bs=4 ga=4  $(date -Iseconds) =====" >> $MASTER

for ARCH in resunet_lc resunet_ma; do
  for STAGE in pretrain finetune; do
    LOG=$LOGDIR/${ARCH}_a3_${STAGE}_round2.log
    echo "===== [$ARCH/$STAGE  bs=4 ga=4] start $(date -Iseconds) =====" > $LOG
    echo "    -> $ARCH $STAGE  bs=4 ga=4" >> $MASTER
    $PY scripts/a3/run_a3_launcher.py --model $ARCH --stage $STAGE \
      --gpus 1 --num-workers 4 --per-gpu-bs 4 --grad-accum 4 >> $LOG 2>&1
    rc=$?
    echo "===== [$ARCH/$STAGE] done rc=$rc $(date -Iseconds) =====" >> $LOG
    echo "    <- $ARCH $STAGE rc=$rc" >> $MASTER
    [ $rc -ne 0 ] && { echo "    chain aborted" >> $MASTER; exit $rc; }
  done
done
echo "===== A100 chain complete $(date -Iseconds) =====" >> $MASTER

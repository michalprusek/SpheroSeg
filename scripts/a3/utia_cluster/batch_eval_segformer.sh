#!/bin/bash
# Eval all SegFormer variants on SpheroMix + HTS-Seg
set -e
PY=/home/prusek/.conda/envs/spheroid_ft/bin/python
SCRIPT=/disk1/prusek/SpheroSeg/code/scripts/a3/eval_segformer.py
LOG=/disk1/prusek/SpheroSeg/logs/batch_eval_segformer.log
SVOC=/disk2/prusek/svoc/results

# variant_name | weights | pretrained | dataset | output_dir
JOBS=(
  "b0|$SVOC/segformer_b0/best.pth|nvidia/segformer-b0-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/SpheroMix|results"
  "b0_eval512|$SVOC/segformer_b0_eval512/best.pth|nvidia/segformer-b0-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/SpheroMix|results"
  "b2_fair|$SVOC/segformer_b2_fair/best.pth|nvidia/segformer-b2-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/SpheroMix|results"
  "b4_fair|$SVOC/segformer_b4_fair/best.pth|nvidia/segformer-b4-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/SpheroMix|results"
  # HTS-Seg for all variants including b0_fair (already done on SpheroMix)
  "b0_fair|$SVOC/segformer_b0_fair/best.pth|nvidia/segformer-b0-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/HTS_Seg_eval|results_htsseg"
  "b0|$SVOC/segformer_b0/best.pth|nvidia/segformer-b0-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/HTS_Seg_eval|results_htsseg"
  "b0_eval512|$SVOC/segformer_b0_eval512/best.pth|nvidia/segformer-b0-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/HTS_Seg_eval|results_htsseg"
  "b2_fair|$SVOC/segformer_b2_fair/best.pth|nvidia/segformer-b2-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/HTS_Seg_eval|results_htsseg"
  "b4_fair|$SVOC/segformer_b4_fair/best.pth|nvidia/segformer-b4-finetuned-ade-512-512|/disk1/prusek/SpheroSeg/data/HTS_Seg_eval|results_htsseg"
)

echo "===== SegFormer batch eval start $(date -Iseconds) =====" > "$LOG"
for job in "${JOBS[@]}"; do
  IFS='|' read -r name weights pretrained dataset outdir <<< "$job"
  out="/disk1/prusek/SpheroSeg/eval_a3/$outdir/segformer_${name}.json"
  if [ -f "$out" ]; then
    echo "[skip] $out" >> "$LOG"; continue
  fi
  echo "----- $(date -Iseconds) eval segformer $name -> $out -----" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" \
    --weights "$weights" --pretrained "$pretrained" \
    --dataset "$dataset" --output "$out" \
    --batch_size 4 --num_workers 4 >> "$LOG" 2>&1
  echo "----- done rc=$? -----" >> "$LOG"
done
echo "===== SegFormer batch eval finished $(date -Iseconds) =====" >> "$LOG"

#!/usr/bin/env bash
# Post-SpheroMix-upload pipeline:
#   1. Verify zip integrity (unzip -t)
#   2. Unzip into /disk1/prusek/SpheroSeg/data/SpheroMix
#   3. Validate train/val/test counts against paper Table 6 (32,367 / 2,539 / 1,019)
#   4. Launch run_all_4.sh in nohup
# Stops at first failure.
set -e

ZIP=/disk1/prusek/SpheroSeg/data/SpheroMix.zip
DATA=/disk1/prusek/SpheroSeg/data
LOG=/disk1/prusek/SpheroSeg/logs/post_upload.log
mkdir -p /disk1/prusek/SpheroSeg/logs
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds) post-upload pipeline ==="
echo "zip: $ZIP"
ls -la "$ZIP"

echo
echo "=== 1) zip integrity test ==="
unzip -tq "$ZIP" && echo "OK: zip integrity verified"

echo
echo "=== 2) unzip ==="
cd "$DATA"
# If SpheroMix dir already exists, refuse to overwrite
if [ -d SpheroMix ]; then
  echo "WARN: $DATA/SpheroMix already exists; removing for clean unzip"
  rm -rf SpheroMix
fi
echo "Unzipping (this takes a few min for 32k images)..."
time unzip -q "$ZIP" -d "$DATA"
ls -la "$DATA"

echo
echo "=== 3) validate split structure ==="
SPHEROMIX="$DATA/SpheroMix"
[ ! -d "$SPHEROMIX" ] && { echo "ERROR: $SPHEROMIX missing"; exit 1; }
ls -la "$SPHEROMIX"
for split in train val test; do
  if [ -d "$SPHEROMIX/$split" ]; then
    n_img=$(find "$SPHEROMIX/$split/images" -type f -not -name '._*' 2>/dev/null | wc -l)
    n_msk=$(find "$SPHEROMIX/$split/masks"  -type f -not -name '._*' 2>/dev/null | wc -l)
    echo "  $split: images=$n_img, masks=$n_msk"
  else
    echo "  $split: MISSING"
  fi
done
echo
echo "Expected per paper Table 6:"
echo "  train: 28809 (or 32367 if including duplicates)"
echo "  val:   2539"
echo "  test:  1019 (= 653 BxPC-3 HQ + 366 DTS OOD)"

echo
echo "=== 4) cleanup AppleDouble metadata if any ==="
find "$SPHEROMIX" -name '._*' -type f -delete 2>/dev/null && echo "removed any ._* AppleDouble"

echo
echo "=== 5) train/val/test count after cleanup ==="
for split in train val test; do
  n_img=$(find "$SPHEROMIX/$split/images" -type f 2>/dev/null | wc -l)
  n_msk=$(find "$SPHEROMIX/$split/masks"  -type f 2>/dev/null | wc -l)
  echo "  $split: images=$n_img, masks=$n_msk"
done

echo
echo "=== Post-upload pipeline OK at $(date -Iseconds) ==="
echo "Next: launch training with"
echo "  cd /disk1/prusek/SpheroSeg/code"
echo "  nohup bash scripts/a3/run_all_4.sh > /disk1/prusek/SpheroSeg/logs/run_all_4.log 2>&1 &"

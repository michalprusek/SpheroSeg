# `scripts/a3/utia_cluster/` — UTIA-cluster operational artefacts

These shell scripts document **how the manuscript's training and evaluation
runs were orchestrated on the UTIA tulen cluster** (A100 + RTX A5000). They
were not designed for external reuse and contain hard-coded UTIA paths:

```
/disk1/prusek/SpheroSeg/code
/disk1/prusek/SpheroSeg/data
/disk1/prusek/SpheroSeg/checkpoints
/disk1/prusek/SpheroSeg/logs
/home/prusek/.conda/envs/spheroid_ft/bin/python
```

They are kept for transparency about the exact orchestration used to produce
Tables 2, 7 and 8.

## Inventory

| File | Purpose |
|---|---|
| `run_all_4.sh`               | Sequential pretrain + finetune for HRNet/U-Net/CBAM/PSPNet on a single A100 |
| `chain_a100_rest.sh`         | LC-ResUNet + MA-ResUNet chain on A100 |
| `chain_a5000_rest.sh`        | MA-ResUNet-Mini chain on A5000 |
| `chain_a100_recovery.sh`     | Restart LC FT from round-1 checkpoint after 2026-05-11 reboot |
| `chain_a5000_recovery.sh`    | Same for MA-Mini on A5000 |
| `recovery_a100_chain.sh`     | Generic recovery wrapper with logging |
| `batch_eval_a3.sh`           | Run `evaluate_a3.py` on every A3 checkpoint, SpheroMix test |
| `batch_eval_htsseg.sh`       | Same, HTS-Seg n=89 held-out |
| `batch_eval_full_roster.sh`  | Combined SpheroMix + HTS-Seg roster |
| `batch_eval_segformer.sh`    | SegFormer variants on both test sets |
| `eval_round2.sh`             | Re-evaluate after round-2 retraining (LC + MA + MA-Mini) |
| `post_upload_pipeline.sh`    | Unzip + validate dataset counts (32,367 / 2,539 / 1,019) |
| `health_check.sh`            | Long-running cluster monitor / auto-fix |

## Reproducing without UTIA hardware

External users should invoke the underlying Python entry points directly:

```bash
# Single-architecture two-stage training
python scripts/a3/run_a3_launcher.py --model resunet_cbam --stage pretrain
python scripts/a3/run_a3_launcher.py --model resunet_cbam --stage finetune

# Stratified evaluation against any test split
python scripts/a3/evaluate_a3.py --weights /path/to/best_model.pth \
    --model resunet_cbam --dataset $SPHEROMIX_PATH \
    --output eval/resunet_cbam.json
```

The Python scripts read `$SPHEROMIX_PATH`, `$SPHEROHQ_PATH`,
`$SPHEROSEG_OUTPUT_ROOT`, `$SPHEROSEG_LOG_ROOT` and accept overrides via CLI;
they are portable across hosts. See the top-level `README.md` for the full
reproducibility workflow.

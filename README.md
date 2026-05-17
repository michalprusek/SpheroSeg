# SpheroSeg: Deep-Learning Models for Tumor Spheroid Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reference implementation and reproducibility package for the manuscript
**"SpheroSeg: Advancing Tumor Spheroid Analysis Through Open-Source Deep Learning"**
(CMPB-D-25-07356, *Computer Methods and Programs in Biomedicine*, Elsevier).
A live deployment of the three production models powers the web application at
[spherosegapp.utia.cas.cz](https://spherosegapp.utia.cas.cz/).

---

## Overview

SpheroSeg trains and benchmarks seven CNN architectures plus three SegFormer
transformer variants on the largest open spheroid bright-field dataset to date.
All architectures follow a uniform two-stage **A3** training protocol so that
performance differences reflect modelling capacity, not hyper-parameter tuning.

Stratified evaluation reports per-image IoU/Dice/precision/recall with
bootstrap 95 % CIs (10 000 iterations) on three held-out test slices:

| Test slice              | n    | Composition                                  |
|---|---|---|
| SpheroHQ (BxPC-3)       | 653  | within-distribution                          |
| DTS                     | 366  | first OOD (cell-line shift)                  |
| HTS-Seg                 | 96   | second OOD (instrument / acquisition shift)  |
| Unified SpheroHQ + DTS  | 1019 | reported alongside slices                    |

## Headline results — A3 protocol (peak per architecture)

| Model               | SpheroHQ IoU         | DTS IoU              | HTS-Seg IoU          |
|---|---|---|---|
| **CBAM-ResUNet**    | 0.9478 [.944, .952] | 0.8781 [.865, .890] | 0.2239 [.201, .248] |
| **U-Net**           | 0.9418 [.937, .946] | 0.9096 [.897, .921] | 0.4205 [.397, .443] |
| **MA-ResUNet**      | 0.9404 [.936, .945] | 0.9068 [.895, .917] | 0.5442 [.521, .566] |
| **PSPNet**          | 0.9395 [.935, .944] | 0.9072 [.894, .919] | 0.4343 [.410, .458] |
| **MA-ResUNet-Mini** | 0.9385 [.934, .943] | 0.8917 [.879, .903] | 0.2579 [.232, .284] |
| **HRNet**           | 0.9361 [.931, .941] | 0.9011 [.888, .913] | 0.2600 [.222, .298] |
| **LC-ResUNet**      | 0.9324 [.926, .938] | 0.9067 [.895, .917] | **0.5507 [.536, .565]** |

The web application ships **U-Net**, **CBAM-ResUNet** and **HRNet** as
production models (215 ms end-to-end on an NVIDIA RTX A5000).

For the SegFormer-B0/B2/B4 transformer baselines (Table 8 of the manuscript)
see [§ SegFormer baselines](#segformer-baselines).

---

## Repository layout

```
SpheroSeg/
├── models/                          # 7 ConvNet architectures + LightM-UNet (legacy)
│   ├── unet.py                      # U-Net
│   ├── hrnet.py                     # HRNet-V2
│   ├── pspnet.py                    # PSPNet (ResNet-101 backbone)
│   ├── resunet_cbam.py              # CBAM-ResUNet
│   ├── resunet_lc.py                # LC-ResUNet (lightweight CBAM)
│   ├── resunet_ma.py                # MA-ResUNet (multi-attention)
│   ├── resunet_ma_mini.py           # MA-ResUNet-Mini
│   └── lightm_unet.py               # LightM-UNet (dropped from manuscript; kept for reference)
│
├── src/
│   ├── training/
│   │   └── CNN_main_spheroid.py     # Core training engine (called by run_a3_launcher)
│   └── inference/
│       ├── inference.py             # Single-image / batch inference
│       └── README_INFERENCE.md
│
├── scripts/
│   ├── a3/                          # A3-protocol reproducibility (see below)
│   │   ├── run_a3_launcher.py       # Standardised launcher (pretrain + finetune)
│   │   ├── evaluate_a3.py           # Stratified evaluator → JSON for Tables 2, 7
│   │   ├── train_segformer.py       # SegFormer fair-protocol trainer (Table 8)
│   │   ├── eval_segformer.py        # SegFormer evaluator
│   │   ├── iaa_analysis.py          # DTS-150 IAA pilot (§5.5.1)
│       ├── generate_sauvola_dts.py  # Sauvola pre-annotation for CVAT
│       ├── build_cvat_package.py    # CVAT task package builder
│       ├── prepare_htsseg_eval.py   # HTS-Seg n=96 eval set prep
│       ├── prepare_htsseg_tiles.py  # HTS-Seg patch-tiled variant
│       └── utia_cluster/            # UTIA-cluster orchestration (archive only)
│           └── *.sh                 # batch_eval_*, chain_*, run_all_4, …
│
├── results/                         # Aggregated per-model results from manuscript
│   ├── evaluation_results_*/
│   ├── statistics/                  # TOPSIS, Friedman/Nemenyi, failure analysis
│   └── app/                         # Web-app inference timing
│
├── model_details/                   # Pre-A3 training configs + logs (archive)
│   ├── README.md                    # explains why this differs from A3
│   └── {model}_{pretrained,finetuned}/
│
├── paper/                           # LaTeX source + bibliography
│
├── docs/PARAMETER_COUNTS.md         # Per-architecture parameter counts
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/michalprusek/SpheroSeg.git
cd SpheroSeg

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.9 – 3.11, CUDA 11.8 / 12.1. SegFormer needs ≥ 5 GB VRAM at
512×512 training; ConvNet pretraining of CBAM/LC-ResUNet at 1024×1024 needs
≥ 16 GB.

---

## Data and weights

| Asset | Where |
|---|---|
| **Datasets** (SpheroHQ 22,683 / SpheroMix 32,367 / DTS / HTS-Seg) | <https://staff.utia.cas.cz/novozada/spheroseg/> |
| **A3 checkpoints** (14 ConvNet + 3 SegFormer, ~7.5 GB) | <https://staff.utia.cas.cz/novozada/spheroseg/> |
| **Web application** (live inference) | <https://spherosegapp.utia.cas.cz> |

After downloading, point the scripts at the dataset roots via environment
variables (all A3 scripts honour these):

```bash
export SPHEROMIX_PATH=/path/to/SpheroMix
export SPHEROHQ_PATH=/path/to/SpheroHQ
export SPHEROSEG_OUTPUT_ROOT=/path/to/checkpoints   # default: ./checkpoints
export SPHEROSEG_LOG_ROOT=/path/to/logs             # default: ./logs
```

Each split is expected to contain matching basenames:

```
<dataset>/
├── train/{images,masks}/
├── val/{images,masks}/
└── test/{images,masks}/
```

Masks are binary PNGs (0 = background, 255 = spheroid).

---

## Reproducing the A3 protocol

The A3 protocol locks every hyper-parameter so that any cross-model gap reflects
architecture, not tuning. The invariants are asserted at runtime by the launcher
and dumped as `a3_protocol_spec.json` next to each checkpoint:

| Setting                | Value |
|---|---|
| effective batch size   | 16 (per-GPU bs × grad-accum × num-GPUs, asserted) |
| optimiser              | AdamW, weight decay 1e-4 |
| scheduler              | OneCycleLR, pct_start 0.1, gradient clip 1.0 |
| peak LR — pretrain     | 2 × 10⁻⁴ |
| peak LR — finetune     | 1 × 10⁻⁵ (encoder frozen 10 epochs) |
| loss                   | focal · 1.0 + dice · 1.0 + IoU · 0.5 (boundary = aux = 0) |
| image size             | 1024 × 1024 |
| epochs                 | 50 (early-stop patience 10, min_delta 1e-4) |
| seed                   | 42 |
| normalisation          | InstanceNorm (`--use_instance_norm`) |
| precision              | mixed (bfloat16 autocast + `GradScaler`) |
| TF32 / cuDNN           | high / benchmark |

### Stage 1 — pretrain on SpheroMix

```bash
python scripts/a3/run_a3_launcher.py \
    --model resunet_cbam \
    --stage pretrain \
    --gpus 1
```

`--per-gpu-bs` and `--grad-accum` are auto-chosen per architecture (see
`MICRO_DEFAULT` in the launcher). Override only if you change `--gpus`.

### Stage 2 — finetune on SpheroHQ

```bash
python scripts/a3/run_a3_launcher.py \
    --model resunet_cbam \
    --stage finetune \
    --gpus 1
```

The finetune stage auto-discovers `<output>/resunet_cbam_a3_pretrained_seed42*/best_model.pth`.
Pass `--pretrained-path /explicit/path.pth` to override.

GPU pinning: set `CUDA_VISIBLE_DEVICES` **before** invoking the script (the
launcher reads it once at import-time):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/a3/run_a3_launcher.py --model unet --stage pretrain
```

### Multi-architecture sweep

```bash
for m in unet hrnet pspnet resunet_cbam resunet_lc resunet_ma resunet_ma_mini; do
    python scripts/a3/run_a3_launcher.py --model $m --stage pretrain
    python scripts/a3/run_a3_launcher.py --model $m --stage finetune
done
```

LightM-UNet was dropped from the published manuscript but the architecture file
and launcher entry remain available for ablation.

### UTIA-cluster orchestration (archive)

Shell scripts under `scripts/a3/utia_cluster/` document the exact orchestration
used to produce the manuscript's runs on the UTIA tulen cluster (A100 + RTX
A5000). They contain hard-coded UTIA paths and are not designed for external
reuse — see `scripts/a3/utia_cluster/README.md` for the inventory. External
users should invoke `run_a3_launcher.py` directly as shown above.

---

## Stratified evaluation

`evaluate_a3.py` runs a single checkpoint over a dataset, emits
per-image rows + bootstrap-CI summaries for the `unified` / `hq_only` /
`dts_only` panels, plus a JSON in the schema consumed by Tables 2, 7 of the
manuscript:

```bash
python scripts/a3/evaluate_a3.py \
    --weights checkpoints/resunet_cbam_a3_finetuned_seed42*/best_model.pth \
    --model   resunet_cbam \
    --dataset $SPHEROMIX_PATH \
    --output  eval_a3/results/resunet_cbam_a3_finetuned.json
```

For HTS-Seg OOD evaluation, point `--dataset` at the HTS-Seg root:

```bash
python scripts/a3/evaluate_a3.py \
    --weights checkpoints/resunet_lc_a3_pretrained_seed42/best_model.pth \
    --model   resunet_lc \
    --dataset /path/to/HTS_Seg_eval_v2 \
    --output  eval_a3/results_htsseg_v2/resunet_lc_a3_pretrained.json
```

Batch wrappers `scripts/a3/batch_eval_*.sh` evaluate every checkpoint in a
directory tree; edit the env-var defaults at the top of each script before use.

---

## SegFormer baselines

The fair-protocol SegFormer trainer (Table 8) shares optimiser, scheduler and
seed with the ConvNet A3 protocol but resizes to 512×512 for training and
upsamples logits to 1024 for evaluation (so encoder receptive fields match the
pre-trained AdE-20k weights):

```bash
# B0
python scripts/a3/train_segformer.py \
    --pretrained nvidia/segformer-b0-finetuned-ade-512-512 \
    --out-dir checkpoints/segformer_b0_fair \
    --epochs 40 --batch-size 8

# B2
python scripts/a3/train_segformer.py \
    --pretrained nvidia/segformer-b2-finetuned-ade-512-512 \
    --out-dir checkpoints/segformer_b2_fair \
    --epochs 40 --batch-size 4 --accum-steps 2

# B4
python scripts/a3/train_segformer.py \
    --pretrained nvidia/segformer-b4-finetuned-ade-512-512 \
    --out-dir checkpoints/segformer_b4_fair \
    --epochs 40 --batch-size 2 --accum-steps 4
```

Evaluate with `scripts/a3/eval_segformer.py` — same JSON schema as
`evaluate_a3.py`, so SegFormer rows merge directly into Table 8.

---

## Inter-annotator agreement pilot (§5.5.1)

`scripts/a3/iaa_analysis.py` reproduces the DTS-150 IAA panel:
pairwise IoU, Fleiss' κ, STAPLE consensus, per-rater-vs-DTS IoU,
object-count Cohen κ, and Bland-Altman area panel.

```bash
python scripts/a3/iaa_analysis.py \
    --raters-root /path/to/cvat_exports \
    --dts-gt-dir  /path/to/dts_original_masks \
    --out-dir     eval_results/iaa
```

Expects each rater's CVAT export under `<raters-root>/a{1,2,3,4}/annotations/instances_default.json`.
Writes `iaa_summary.json` (drop-in for §5.5.1), `iaa_pairwise_iou.csv`,
`iaa_heatmap.pdf`, and `iaa_bland_altman_area.pdf`.

---

## Inference

```bash
python src/inference/inference.py \
    --model    resunet_cbam \
    --weights  checkpoints/resunet_cbam_a3_finetuned_seed42_a5000/best_model.pth \
    --input    /path/to/images \
    --output   /path/to/predictions \
    --use-tta \
    --save-overlay
```

The CLI also drives the same code path as the production web application.

---

## Statistical analysis

```bash
python results/statistics/comprehensive_analysis.py
```

Produces:

- `model_summary_statistics.csv` — per-model means + bootstrap CIs
- `topsis_rankings.csv` — multi-criteria TOPSIS ranking
- `failure_analysis.csv` — cases with IoU < 0.7
- `tests.csv` — Friedman + Nemenyi posthoc

---

## Troubleshooting

**Out of memory.** Reduce `--per-gpu-bs` and increase `--grad-accum`
proportionally — the launcher will refuse to start if `bs × ga × gpus ≠ 16`.

**`ModuleNotFoundError: einops` (or `transformers`).** Reinstall from the
updated `requirements.txt`; `einops` is needed for LightM-UNet, `transformers`
for SegFormer.

**Hard-coded `/disk1/prusek/...` paths.** All scripts now derive `REPO_ROOT`
from `__file__` and accept dataset paths via env vars / CLI flags. If you
still see absolute UTIA paths the script needs an `export SPHEROMIX_PATH=...`
or `--dataset` flag.

---

## Citation

If you use this code, the trained weights, or the SpheroHQ / SpheroMix
datasets, please cite:

```bibtex
@article{prusek2026spheroseg,
  title   = {SpheroSeg: Advancing Tumor Spheroid Analysis Through Open-Source Deep Learning},
  author  = {Pr{\\u{u}}{\\v{s}}ek, Michal and others},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2026},
  note    = {Under major revision (manuscript CMPB-D-25-07356)}
}
```

---

## Contact

- **Michal Průšek** (first author) — prusemic@cvut.cz
- **Adam Novozámský** (corresponding author, UTIA CAS) — novozamsky@utia.cas.cz
- Bug reports / feature requests: [GitHub Issues](https://github.com/michalprusek/SpheroSeg/issues)

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Institute of Information Theory and Automation, Czech Academy of Sciences (ÚTIA)
- University of Chemistry and Technology Prague (VŠCHT / UCT)
- All annotators who contributed to the SpheroHQ dataset and the DTS-150 IAA pilot.

#!/usr/bin/env python
"""
A3 standardized-protocol launcher (pretrain + finetune).

Locked invariants (both stages):
  effective batch size = 16  (per_gpu_bs * grad_accum * num_gpus, asserted at runtime)
  loss = focal(1.0) + dice(1.0) + iou(0.5),  boundary=0, aux=0
  img_size = 1024, patience 10, min_delta 1e-4
  AdamW + OneCycleLR, weight_decay = 1e-4
  seed = 42, cudnn.benchmark, TF32 high

Stage differences:
  pretrain : SpheroMix dataset, LR 2e-4, max 50 epochs, no frozen encoder
  finetune : SpheroHQ  dataset, LR 1e-5, max 50 epochs, freeze encoder 10 epochs,
             requires --pretrained-path or auto-finds it from pretrain output dir

Use shell env CUDA_VISIBLE_DEVICES to pick GPU (do NOT use a CLI flag — torch
locks CUDA_VISIBLE_DEVICES at import time, so it must be set before python starts).
"""
import os, sys, random, argparse, json
from pathlib import Path

# === LOCKED PROTOCOL CONSTANTS (both stages) ======================
SEED              = 42
EPOCHS            = 50
WEIGHT_DECAY      = 1e-4
IMG_SIZE          = 1024
EFFECTIVE_BATCH   = 16
PATIENCE          = 10
MIN_DELTA         = 1e-4
GRADIENT_CLIP_VAL = 1.0
FOCAL_W   = 1.0
DICE_W    = 1.0
IOU_W     = 0.5
BOUNDARY_W= 0.0
AUX_W     = 0.0

# Stage-specific
STAGE_CFG = {
    "pretrain": {
        "dataset": "/disk1/prusek/SpheroSeg/data/SpheroMix",
        "lr": 2e-4,
        "freeze_backbone_epochs": 0,
        "out_tag": "pretrained",
    },
    "finetune": {
        "dataset": "/disk1/prusek/SpheroSeg/data/SpheroHQ",
        "lr": 1e-5,
        "freeze_backbone_epochs": 10,    # standardized; originals were 4–15
        "out_tag": "finetuned",
    },
}

MICRO_DEFAULT = {
    "unet":            {"bs": 8, "ga": 2},
    "hrnet":           {"bs": 8, "ga": 2},
    "resunet_cbam":    {"bs": 4, "ga": 4},
    "pspnet":          {"bs": 4, "ga": 4},
    "lightm_unet":     {"bs": 4, "ga": 4},
    "resunet_ma":      {"bs": 4, "ga": 4},
    "resunet_ma_mini": {"bs": 8, "ga": 2},
    "resunet_lc":      {"bs": 4, "ga": 4},
}
for _m, _cfg in MICRO_DEFAULT.items():
    assert _cfg["bs"] * _cfg["ga"] == EFFECTIVE_BATCH, _m

OUTPUT_ROOT = "/disk1/prusek/SpheroSeg/checkpoints"
LOG_ROOT    = "/disk1/prusek/SpheroSeg/logs"


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def find_pretrained_best(model_key: str, suffix: str) -> Path:
    """Auto-discover pretrain output for finetune."""
    suf = f"_{suffix}" if suffix else ""
    cands = [
        Path(f"{OUTPUT_ROOT}/{model_key}_a3_pretrained_seed{SEED}{suf}/best_model.pth"),
        Path(f"{OUTPUT_ROOT}/{model_key}_a3_pretrained_seed{SEED}/best_model.pth"),  # fallback (no suffix)
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No pretrain best_model.pth for model={model_key} (looked at {cands})")


def main():
    parser = argparse.ArgumentParser(description="A3 standardized-protocol launcher")
    parser.add_argument("--model", required=True, choices=list(MICRO_DEFAULT.keys()))
    parser.add_argument("--stage", required=True, choices=list(STAGE_CFG.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--per-gpu-bs", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--pretrained-path", default=None,
                        help="Required for --stage finetune; if omitted, auto-discovered "
                             "from {model}_a3_pretrained_seed42[_suffix]/best_model.pth")
    a = parser.parse_args()

    set_seeds(SEED)

    stage_cfg = STAGE_CFG[a.stage]

    micro = dict(MICRO_DEFAULT[a.model])
    if a.per_gpu_bs is not None: micro["bs"] = a.per_gpu_bs
    if a.grad_accum is not None: micro["ga"] = a.grad_accum
    per_gpu_bs = micro["bs"]
    grad_accum = micro["ga"]
    num_gpus   = a.gpus

    effective = per_gpu_bs * grad_accum * num_gpus
    assert effective == EFFECTIVE_BATCH, (
        f"\nEFFECTIVE BATCH SIZE INVARIANT VIOLATED for {a.model}/{a.stage}\n"
        f"  per_gpu_bs={per_gpu_bs} * grad_accum={grad_accum} * num_gpus={num_gpus} = {effective}\n"
        f"  expected = {EFFECTIVE_BATCH}")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"[a3/{a.stage}] effective batch invariant OK: "
          f"per_gpu={per_gpu_bs} grad_accum={grad_accum} gpus={num_gpus}  → eff={effective}")
    print(f"[a3/{a.stage}] CUDA_VISIBLE_DEVICES={cvd}")

    # Resolve pretrained path for finetune
    pretrained_path = a.pretrained_path
    if a.stage == "finetune" and pretrained_path is None:
        pretrained_path = str(find_pretrained_best(a.model, a.output_suffix))
        print(f"[a3/finetune] auto-discovered pretrained_path: {pretrained_path}")

    suffix = f"_{a.output_suffix}" if a.output_suffix else ""
    output_dir = f"{OUTPUT_ROOT}/{a.model}_a3_{stage_cfg['out_tag']}_seed{SEED}{suffix}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(LOG_ROOT).mkdir(parents=True, exist_ok=True)

    argv = [
        "CNN_main_spheroid.py",
        "--dataset_path",   stage_cfg["dataset"],
        "--output_dir",     output_dir,
        "--model",          a.model,
        "--epochs",         str(EPOCHS),
        "--batch_size",     str(per_gpu_bs),
        "--lr",             str(stage_cfg["lr"]),
        "--weight_decay",   str(WEIGHT_DECAY),
        "--img_size",       str(IMG_SIZE),
        "--focal_weight",   str(FOCAL_W),
        "--dice_weight",    str(DICE_W),
        "--iou_weight",     str(IOU_W),
        "--boundary_weight",str(BOUNDARY_W),
        "--aux_weight",     str(AUX_W),
        "--optimizer",      "adamw",
        "--scheduler",      "onecycle",
        "--num_workers",    str(a.num_workers),
        "--patience",       str(PATIENCE),
        "--gpus",           str(num_gpus),
        "--use_instance_norm",
        "--min_delta",      str(MIN_DELTA),
        "--use_cache",
        "--gradient_accumulation_steps", str(grad_accum),
        "--gradient_clip_val", str(GRADIENT_CLIP_VAL),
        "--freeze_backbone_epochs", str(stage_cfg["freeze_backbone_epochs"]),
    ]
    if pretrained_path:
        argv += ["--pretrained_path", pretrained_path]

    spec_dump = {
        "model": a.model, "stage": a.stage, "seed": SEED,
        "effective_batch_size": EFFECTIVE_BATCH, "effective_batch_actual": effective,
        "per_gpu_bs": per_gpu_bs, "grad_accum": grad_accum, "num_gpus": num_gpus,
        "epochs": EPOCHS, "lr": stage_cfg["lr"], "weight_decay": WEIGHT_DECAY,
        "img_size": IMG_SIZE,
        "loss_weights": {"focal": FOCAL_W, "dice": DICE_W, "iou": IOU_W,
                         "boundary": BOUNDARY_W, "aux": AUX_W},
        "scheduler": "onecycle", "patience": PATIENCE, "min_delta": MIN_DELTA,
        "gradient_clip_val": GRADIENT_CLIP_VAL,
        "freeze_backbone_epochs": stage_cfg["freeze_backbone_epochs"],
        "pretrained_path": pretrained_path,
        "dataset_path": stage_cfg["dataset"],
        "cuda_visible_devices": cvd, "output_suffix": a.output_suffix,
        "argv": argv,
    }
    with open(f"{output_dir}/a3_protocol_spec.json", "w") as f:
        json.dump(spec_dump, f, indent=2)
    print(f"[a3/{a.stage}] audit trail → {output_dir}/a3_protocol_spec.json")

    if a.dry_run:
        print(f"[a3/{a.stage}] dry-run; argv:\n  " + " ".join(argv))
        return

    sys.argv = argv
    repo_root = Path("/disk1/prusek/SpheroSeg/code").resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src" / "training"))

    print(f"[a3/{a.stage}] launching CNN_main_spheroid.main() for model={a.model}")
    sys.stdout.flush()

    from CNN_main_spheroid import main as orig_main
    orig_main()

if __name__ == "__main__":
    main()

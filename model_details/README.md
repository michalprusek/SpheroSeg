# `model_details/` — pre-revision archive

This directory contains the original (pre-A3) training configurations and logs
for the **first** submission of CMPB-D-25-07356. Each subdirectory holds the
`config.json` and `training.log` captured during that early sweep.

These configs do **not** match the standardised **A3 protocol** introduced
during the major revision (reviewer #1, issue 3 — training-protocol
inconsistency). Differences include:

| Setting               | Pre-A3 (this archive)            | A3 (revision)                      |
|---|---|---|
| epochs                | 100 – 150 (varies)               | 50 (fixed)                          |
| scheduler             | `cosine` (varies)                | `OneCycleLR` (fixed)                |
| effective batch size  | 12 – 40 (varies)                 | 16 (asserted at runtime)            |
| patience              | 20 – 25 (varies)                 | 10 (fixed)                          |
| loss boundary weight  | 0.1 (varies)                     | 0.0 (fixed)                         |
| seed                  | mixed                            | 42 (fixed)                          |

The current Tables 2, 7 and 8 of the revised manuscript are produced by the
A3 protocol. For each A3 checkpoint, the actual hyper-parameters used are
captured in an `a3_protocol_spec.json` written next to `best_model.pth` by
`scripts/a3/run_a3_launcher.py`.

This archive is preserved for historical reproducibility of the pre-revision
results only; do **not** use these configs to reproduce the revised tables.

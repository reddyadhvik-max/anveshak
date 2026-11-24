# Phase 1: Pipeline + baseline (2026-11-18 → 2026-11-21)

## Goal
Get the full pipeline running on the training machine (Lenovo LOQ, RTX 5050 Laptop 8 GB, native Windows 11),
measure which model scale fits the Stage A budget, and make long runs safe to pause, stop and resume.

## What was built
| File | Purpose |
|---|---|
| `anveshak/control.py` | Local control panel (localhost:8790): per-run state, epoch log with metrics and ETA, Pause (after the epoch's checkpoint), Stop now (within one batch, resumable), Resume (`--resume last.pt`) |
| `anveshak/train.py` | `--detach` via `spawn()` (Windows: hidden console, breaks away from the shell's job), STOP / PAUSE flags checked per batch / per checkpoint |
| `scripts/loq/setup_windows.ps1`, `open_panel.pyw` | One-shot LOQ setup (cu128 PyTorch for sm_120, data, convert) and a desktop shortcut for the panel |
| `splits/` | Frozen val splits (A 952 tiles, B 172), identical on every machine |
| `anveshak/infer.py` | Our own post-processing over the raw 52-slot head at native 1024, FP32: multi-label per-class NMS for known slots, unknown = max(s_unk, s_obj · (1 − max known)), the competition's conflict rule applied by us |
| `tools/rfs.py` | Repeat-factor sampling (LVIS) as an oversampled train list: epoch ×1.30, rarest class 46 → 204 tiles, imbalance 102:1 → 34:1 |
| `anveshak/patch.py` | `ASSIGN_LIMIT`: dense batches are assigned per image before the GPU spills (see below) |

## Scale benchmark (details: `PHASE_1_bench.md`)
| Round | Finding |
|---|---|
| 1 | Every config spilled past 8 GB into Windows shared memory (WDDM sysmem fallback): about 30 W, slow, RAM at 0 GB free. The peak came from the task-aligned assigner on dense tiles (up to 2,102 boxes), not from the batch size. |
| fix | NVIDIA "CUDA Sysmem Fallback Policy = Prefer No Sysmem Fallback" (so a real OOM is raised) + `patch.ASSIGN_LIMIT` (assign per image up front on dense batches). Tested: per-image assignment equals the batched one. |
| 3 | s b8 and m b4 hit a real OOM in epoch 1 and were halved to **s b4 / m b2**, which ran clean (62–68 W, shared memory < 1 GB). Epochs in 125 GPU-h with RFS: s b4 ≈ 570, m b2 ≈ 325. |

## Key decisions
- **Scale m at batch 2** (≈ 325 epochs fit; accuracy first). The batch is passed explicitly, because Ultralytics only halves it on an OOM in the first epoch.
- **Stage A trains on the RFS list** (`data_A_rfs.yaml`).
- **The panel is the only way to pause / stop:** flags on disk, no signals, so it works the same on Windows and macOS. Every pause and resume is logged in `provenance.json`.
- **Insurance CSV:** the first validator-passing test CSV came from Stage A itself (`insurance_A.csv`, 612,810 rows, 23 Nov; see Phase 2) instead of a short throwaway run, to keep the GPU on the main lineage.

## Verification
- Panel button test on Windows: pause → resume → stop in 3 s → resume → finished; a detached run survives its shell.
- `infer.py` on 20 real tiles passes the official validator.
- 16/16 tests on the LOQ at the end of the phase.

## Next
Phase 2: Stage A from random init, m, batch 2, 250 epochs.

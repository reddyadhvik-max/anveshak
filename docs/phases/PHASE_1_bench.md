# Phase 1: scale benchmark on the LOQ (s vs m)

Date: 2026-11-18. Machine: Lenovo LOQ 15AHP10, Windows 11 native, RTX 5050 Laptop 8 GB (driver 595.95),
16 GB RAM. torch 2.11.0+cu128, Ultralytics 8.4.215. Model: `configs/yolo11-p2.yaml` (52 slots), random init.

Protocol: `python -m anveshak.train --data data_A.yaml epochs=1 fraction=0.1 imgsz=1024 nbs=64 optimizer=SGD plots=False`
(871 training tiles, then validation on the frozen 952-tile val split). GPU sampled every 5 s with `nvidia-smi`;
Windows GPU shared-memory and free RAM sampled every 5 s with performance counters (second round only).

## Rounds 1–2 (before the fix; spill-bound)

| config | train, 10 % | full epoch | × 1.30 (RFS) | epochs in 125 GPU-h | val (952 tiles) | Ultralytics GPU_mem | shared spill | mean / max power | max temp | assigner OOM retries |
|---|---|---|---|---|---|---|---|---|---|---|
| s, batch 8, workers 6 | 6:12 | 62 min | 81 min | **≈ 93** | 15 s | 12.6 G | yes | 30 / 70 W | 63 °C | 2 |
| m, batch 4, workers 6 | 7:18 | 73 min | 95 min | **≈ 79** | 32 s | 13.4 G | yes | 32 / 70 W | 66 °C | 2 |
| s, batch 4, workers 4 | 2:29 | 25 min | 32 min | **≈ 232** | 16 s | 13.4 G | 7.7 GB peak | 31 / 70 W | 63 °C | 1 |
| m, batch 2, workers 4 | 1:49 | 18 min | 24 min | **≈ 317** | 48 s | 10.2 G | 6.6 GB peak | 56 / 71 W | 69 °C | 1 |

Epoch budget = 125 h ÷ (RFS epoch + val). The once-per-launch overhead (startup, the final `best.pt` evaluation)
added 4–7 min of wall time per run and is not included. `rfs.py`: 8707 tiles → 11347 samples per epoch (× 1.30).

## What the numbers say

- **Every config spills past the 8 GB of VRAM.** Dedicated VRAM sits pinned at 7.82–7.87 GB in every run, and the
  Windows driver's *sysmem fallback* puts the rest in shared system memory (up to 7.7 GB). Ultralytics' `GPU_mem`
  (10–13 GB) counts both. Halving the batch barely lowered it: the peaks come from the TaskAlignedAssigner on dense
  tiles (up to 1202 objects per tile against ~87k anchors with the P2 head at 1024), not from the batch size.
- **The spill dominates speed, so these timings are noisy.** m at batch 2 beat s at batch 4, which can't be true in
  steady state, since m costs about 3× the compute. Power is the tell: about 30 W while spilling, 56 W for the m b2 run.
  Thermals are fine (max 69 °C, the only throttle reason is 0x4, the software power cap).
- **System RAM hit 0 GB free** during the second round (16 GB total; shared GPU memory, 4–6 dataloader workers
  and the main process together). In the first round Claude Code's own memory guard killed the benchmark shell.
  A multi-day run like this risks the same thing.
- No "TaskAlignedAssigner … CPU" fallback. The few OOMs were retried one image at a time on the GPU.
- The workers 4 vs 8 comparison was not completed (the run was killed by the memory guard). Given RAM, use workers ≤ 4.
- `infer.py` sanity check (20 val tiles, `bench_s` last.pt, FP32, GPU): 6 s wall including model load, 0 rows
  (expected after 1 epoch), and the official validator passes. Note: `--device 0` fails (`Invalid device string`); use `--device cuda`.

## Round 3: after the fix (sysmem fallback off and patch.py per-image assignment, 684aa35)

The user set NVIDIA **CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback**, and `patch.ASSIGN_LIMIT` makes dense
batches take the per-image assigner path up front. Requested: s b8 and m b4, workers 4.

**Both requested batches hit a real CUDA OOM in their first epoch, and Ultralytics halved the batch automatically**
(`trainer.py`: *"CUDA out of memory with batch=8. Reducing to batch=4 and retrying (1/3)"*; for m, batch=4 → 2).
The OOM was raised inside the trainer's forward + loss + backward block, so not in the assigner, which logged its own
per-image retries separately. For s it came right after an assigner per-image retry (max_num_obj 641). For m it came
right at the start of the epoch. So the effective configs, and the clean timings, are **s b4** and **m b2**:

| effective config | train, 10 % | full epoch | × 1.30 (RFS) + val | epochs in 125 GPU-h | val | dedicated VRAM peak | shared spill peak | mean / max power | max temp | RAM free min / mean |
|---|---|---|---|---|---|---|---|---|---|---|
| s b4 (asked for b8) | 0:59.7 | 10.0 min | 13.2 min | **≈ 570** | 15.7 s | 7.73 GB | 0.9 GB | 68 / 75 W | 67 °C | 0.0 / 2.0 GB |
| m b2 (asked for b4) | 1:43 | 17.2 min | 23.1 min | **≈ 325** | 47.7 s | 7.58 GB | 0.5 GB | 62 / 70 W | 69 °C | 0.2 / 1.8 GB |

- The fix works. s b4 went from 2:29 to 1:00 (2.5× faster), and power went from about 30 W to 62–68 W. Shared memory is now
  under 1 GB, close to the baseline Windows keeps there anyway. m b2 barely changed (1:49 → 1:43), so it had been spilling only a little.
- **Ultralytics only auto-halves in the first epoch** (`epoch > start_epoch → raise`). A later OOM crashes the run
  (resumable from `last.pt`). So set the batch that ran clean: **s batch=4, m batch=2**. Don't ask for b8 / b4 and
  rely on halving: a resume starts at a later epoch, so it would crash.
- RAM is still tight (min 0–0.2 GB free, mean about 2 GB of 16). Keep workers=4 and avoid heavy apps during training.

## Recommendation

The rule is "the largest scale that fits ≥ 150 epochs in about 125 GPU-hours". **m at batch=2 fits about 325 epochs**, more
than double the target, so **choose m, batch=2, nbs=64** (gradient accumulation of 32 keeps the effective batch at 64),
workers=4, on `data_A_rfs.yaml`. s b4 (about 570 epochs) is the fallback if m's quality or stability disappoints.
Larger scales were not tested.

## Earlier provisional recommendation (rounds 1–2, superseded)

(Superseded by round 3.) On these numbers **m at batch 2 fits
(≈ 317 epochs), and so does s at batch 4 (≈ 232)**. But the spill makes both numbers unreliable, and RAM is
exhausted. Before choosing, fix the spill and re-bench (about 30 min):

1. NVIDIA Control Panel → Manage 3D settings → **CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback**
   (for `.venv\Scripts\python.exe`, or globally). Real OOMs then reach PyTorch, and Ultralytics' assigner retries
   that image instead of silently paging. This needs the user (GUI).
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for training (less fragmentation at the peaks).
3. Re-run m b2 and s b4 in swapped order, workers=4, and check that power stays above 50 W with no shared spill.
   If m b2 then fits at ≥ 150 epochs, choose **m**. Otherwise choose s b4.

# Phase 1: scale benchmark on the LOQ (s vs m)

Date: 2026-11-18. Machine: Lenovo LOQ 15AHP10, Windows 11 native, RTX 5050 Laptop 8 GB (driver 595.95),
16 GB RAM. torch 2.11.0+cu128, Ultralytics 8.4.215. Model: `configs/yolo11-p2.yaml` (52 slots), random init.

Protocol: `python -m anveshak.train --data data_A.yaml epochs=1 fraction=0.1 imgsz=1024 nbs=64 optimizer=SGD plots=False`
(871 training tiles, then validation on the frozen 952-tile val split). GPU sampled every 5 s with `nvidia-smi`;
Windows GPU shared-memory and free RAM sampled every 5 s with performance counters (second round only).

## Results

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

## Recommendation (provisional)

The rule is "the largest scale that fits ≥ 150 epochs in about 125 GPU-hours". On these numbers **m at batch 2 fits
(≈ 317 epochs), and so does s at batch 4 (≈ 232)**. But the spill makes both numbers unreliable, and RAM is
exhausted. Before choosing, fix the spill and re-bench (about 30 min):

1. NVIDIA Control Panel → Manage 3D settings → **CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback**
   (for `.venv\Scripts\python.exe`, or globally). Real OOMs then reach PyTorch, and Ultralytics' assigner retries
   that image instead of silently paging. This needs the user (GUI).
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for training (less fragmentation at the peaks).
3. Re-run m b2 and s b4 in swapped order, workers=4, and check that power stays above 50 W with no shared spill.
   If m b2 then fits at ≥ 150 epochs, choose **m**. Otherwise choose s b4.

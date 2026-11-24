# Phase 2: Base detector, Stage A (2026-11-21 → 2026-11-23)

## Goal
Train the first stage of the single lineage from random init on Set A (30 base classes), with the objectness slot,
and get the most accuracy the ~125 GPU-h budget allows.

## Recipe
YOLO11-m + P2 head (`configs/yolo11-p2.yaml`, nc 52), random init (sha256 in `runs/A/provenance.json`), `data_A_rfs.yaml`,
250 epochs, imgsz 1024 native, batch 2 (nbs 64 accumulation), SGD lr0 0.01 / lrf 0.01, mosaic (close_mosaic 10),
`cls=1.0`, flipud 0.5, max_det 1000, deterministic, seed 0.

## Fix during the phase: chunked assignment
The first launch hit a real OOM on a batch with ~1,100 boxes and Ultralytics dropped to batch 1 for the whole run.
`patch.chunked_assign` runs task-aligned assignment 128 GT boxes at a time with identical output (tested against the
original on 300 boxes / 561 multi-positive anchors). A stress test on the 64 densest tiles then peaked at 4.8 GB at
batch 2 with no halving. The batch-1 run was kept as `runs/A_b1_aborted`; Stage A was relaunched at batch 2.

## Results (A-val, 952 tiles)
| Epoch | mAP50 | mAP50-95 |
|---|---|---|
| 25 | 0.125 | |
| 100 | 0.240 | |
| 150 | 0.253 | 0.138 |
| 225 | 0.278 | |
| **250 (best)** | **0.285** | **0.155** |

P 0.496, R 0.331. ~21.6 min/epoch, 11 user pauses/resumes (all in `provenance.json`). `best.pt` sha256 `59b5ea14…d5d6`.
With our own post-processing and scorer: **A-val mAP_K 0.2497** (macro AP50 over the 30 base classes).

## Observations that shaped the next phases
- The curve plateaus from about epoch 100; the last 0.03 came from the LR decay and close_mosaic.
- Per class (from `oe_score --per-class`, 24 Nov): the two giant classes (≈ 72k of 80k val boxes) are at 0.70 / 0.78 AP,
  while 15 of 30 classes are below 0.1. `tools.confusion` later showed why: unknown boxes deleting correct known boxes
  through the conflict rule, confusion of fine-grained classes with the two giants, and two large-object classes mostly
  missed. These are handled in Phases 3–4.

## Next
Phase 3 (open world) and Phase 4 (few-shot) start from `runs/A/weights/best.pt`.

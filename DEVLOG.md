# Development log

One row per phase. Details are in `docs/phases/PHASE_<n>.md`. Each phase is a `--no-ff` merge on `main` with an annotated tag.

| Phase | Tag | Date | What changed | A-val mAP_K (base) | B-val mAP (few-shot) | Dev-sim U-F1 | Dev-sim OE |
|---|---|---|---|---|---|---|---|
| P0 Foundations | `p0-foundations` | 2026-11-18 | Data audit, local OE scorer, validator-safe CSV writer, from-scratch guards, fixed 52-slot P2 model, objectness slot, stage runner | n/a | n/a | n/a | n/a |
| P1 Pipeline + baseline | `p1-baseline` | 2026-11-21 | LOQ setup (native Windows), pause/stop/resume panel, `infer.py` open-world post-processing, RFS, scale benchmark + GPU-spill fix → m batch 2 | n/a | n/a | n/a | n/a |
| P2 Base detector | `p2-base-detector` | 2026-11-23 | Stage A from random init: m-P2, 250 epochs, batch 2, chunked assigner; val mAP50 0.285 / mAP50-95 0.155 | 0.2497 | n/a | n/a | n/a |

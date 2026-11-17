# Phase 0: Foundations (2026-11-18)

## Goal
Lay the base everything else stands on: know the data exactly, measure the real metric locally,
never produce an invalid CSV, make it impossible to use pretrained weights by accident, and fix the
model architecture for the whole lineage.

## What was built
| File | Purpose |
|---|---|
| `tools/eda.py` | Data audit (tables below) |
| `tools/convert.py` | JSON → YOLO labels, deterministic stratified val split (frozen in `splits/` once created on the full data), `data_A.yaml` / `data_B.yaml` |
| `tools/oe_score.py` | Re-implementation of the published OE-Score: conflict rule → VOC all-point mAP_K → U-Rec/U-Prec/U-F1 → OE. Supports `--as-unknown` to turn held-out classes into unknown GT (dev sim) |
| `tools/make_csv.py` | Clips, rounds, dedupes, caps at 1000 rows per tile, then runs the official `validate_predictions.py` |
| `configs/yolo11-p2.yaml` | YOLO11 + P2 head, **nc = 52 fixed**: 0–39 R1 ids, 40–49 reserved R2, 50 unknown, 51 objectness |
| `anveshak/patch.py` | (1) disables the AMP check that downloads `yolo11n.pt`; (2) objectness slot: target[51] = max(target[0..50]) for every positive anchor |
| `anveshak/train.py` | Stage runner. First stage from yaml (random init), later stages `--from` the previous checkpoint. Writes `provenance.json` |
| `tests/` | 12 tests: metric edge cases (perfect, no unknowns → OE 0, unknown flood, conflict ties, duplicates, class skipping), CSV vs the official validator, objectness target, AMP patch, init hashing |

## Verification
- `pytest -q tests` → 12 passed.
- CPU smoke run on real tiles, offline (`YOLO_OFFLINE=1`): random-init stage A → stage B continued from A's checkpoint with `freeze=10`. No `.pt` file was downloaded anywhere. Provenance: init sha256 `d2ca0c30…` (seed 0, scale n), B started from `c4ddf350…`.

## Key decisions
- **Classes:** `class_list.csv` = ids 0–39 + `unknown` = 50, so R1 has 40 classes (Set A ids 0–29 base, Set B ids 30–39 few-shot), and 40–49 are left for R2. Our slots equal the organisers' ids.
- **P2 head, native 1024:** 61% of boxes are < 32 px and 35% < 16 px.
- **Fixed head size:** no layer ever changes shape, so the lineage stays one architecture.
- **Objectness slot inside the head:** a class-agnostic "is this an object" signal learned from every labelled box. It's the base for flagging unknowns, with no second model.
- **`cls=1.0`:** the objectness slot doubles the target-score normaliser, so this restores the class-loss scale.
- **Set B needs old-class pseudo-labels:** the median Set B tile has 0 labels but surely contains base objects.

## Data audit
### dataset_A

- tiles: 9659 (3532 with no labels)
- boxes: 772752; per tile median 5, p90 275, max 2102
- box size (sqrt area, px) p5/p25/p50/p75/p95: [10, 14, 24, 43, 83]
- under 16 px: 35.4%, under 32 px: 60.8%

| class_id | boxes | tiles |
|---:|---:|---:|
| 0 | 698 | 304 |
| 1 | 707 | 180 |
| 2 | 671 | 324 |
| 3 | 789 | 185 |
| 4 | 953 | 149 |
| 5 | 710 | 52 |
| 6 | 864 | 396 |
| 7 | 1135 | 241 |
| 8 | 1113 | 608 |
| 9 | 825 | 287 |
| 10 | 1463 | 578 |
| 11 | 1231 | 488 |
| 12 | 669 | 193 |
| 13 | 1384 | 421 |
| 14 | 1506 | 529 |
| 15 | 1631 | 209 |
| 16 | 1435 | 352 |
| 17 | 1236 | 290 |
| 18 | 844 | 82 |
| 19 | 2110 | 99 |
| 20 | 3876 | 544 |
| 21 | 4706 | 821 |
| 22 | 4403 | 1310 |
| 23 | 4407 | 728 |
| 24 | 4659 | 1231 |
| 25 | 7194 | 1604 |
| 26 | 8926 | 1936 |
| 27 | 14181 | 2505 |
| 28 | 265677 | 4604 |
| 29 | 432749 | 5197 |

### dataset_B

- tiles: 677 (480 with no labels)
- boxes: 428; per tile median 0, p90 2, max 15
- box size (sqrt area, px) p5/p25/p50/p75/p95: [16, 22, 35, 73, 186]
- under 16 px: 5.4%, under 32 px: 46.3%

| class_id | boxes | tiles |
|---:|---:|---:|
| 30 | 16 | 16 |
| 31 | 34 | 25 |
| 32 | 34 | 27 |
| 33 | 27 | 24 |
| 34 | 15 | 15 |
| 35 | 40 | 22 |
| 36 | 79 | 38 |
| 37 | 62 | 21 |
| 38 | 53 | 32 |
| 39 | 68 | 47 |


## Next: Phase 1
LOQ setup (WSL2, CUDA 12.8 torch), full data, `convert.py` on the real data (creates and freezes `splits/`), repeat-factor sampling, `infer.py` v1, s vs m benchmark at 1024, insurance baseline → first validator-passing CSV.

# Anveshak: open-world object detection for remote sensing

Solo entry for Techfest 2026-27 Anveshak (IITB-HRTI). One YOLO11-P2 detector, **trained from random
initialisation**, that detects the known classes, flags everything else as `unknown`, and learns new
classes from a few images without forgetting. The model is a single lineage: Set A → Set B → Round 2.

- Method and schedule: [PLAN.md](PLAN.md)
- Development history, phase by phase: [DEVLOG.md](DEVLOG.md) and [docs/phases/](docs/phases/)
- `git log --graph --oneline --decorate` shows one merge and one tag per phase.

## No pretrained weights
- The model is always built from [`configs/yolo11-p2.yaml`](configs/yolo11-p2.yaml), never from a `.pt`.
- [`anveshak/patch.py`](anveshak/patch.py) disables Ultralytics' AMP check, which would otherwise download `yolo11n.pt`.
- Every stage writes `runs/<stage>/provenance.json`: the sha256 of the random init (first stage) or of the checkpoint it continued from (later stages).

## Setup
```bash
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx; plain `torch` on CPU/Mac
uv pip install -r requirements.txt
pytest -q tests
```

## Data
Only the competition data is used: round1 Dataset A, Dataset B and the test tiles (inference only). No external data.
```bash
python tools/convert.py --root ~/data/round1      # YOLO labels + frozen split (splits/) + data_A/B.yaml
python tools/eda.py --root ~/data/round1          # audit
```

## Train (one lineage)
Every Round 1 step, in order, is in [`configs/pipeline_r1.yaml`](configs/pipeline_r1.yaml). One command runs them
back to back and skips finished steps (`runs/pipeline/done.json`):
```bash
python -m anveshak.pipeline --detach          # or Resume in the panel
```
The lineage that produces the submission (each stage starts from the previous checkpoint; `--from` checks that all
803 tensors match it before training):

| Stage | From | Data | Recipe | A-val mAP_K |
|---|---|---|---|---|
| A | random init | Set A train, repeat-factor sampling (RFS thr 0.1) | 1024 px, batch 2, 250 epochs | 0.296 |
| A2 → A3 → A4 | previous stage | same | native 640-px windows of each 1024 tile (`tile=True`), batch 8, scale 0.25, 12 epochs each | 0.350 → 0.379 → 0.404 |
| **B_final3** (R1) | A4 | Set B train + val ×6 (base classes pseudo-labelled by A4, weak ones greyed out) + 6000 Set A replay tiles + 1500 Set A tiles with pasted Set B objects | A2's recipe, backbone frozen (10 layers), lr0 0.0005, 25 epochs | 0.394 |

The R1 checkpoint is chosen by estimated OE, not by mAP_K alone. Each extra Set A round sharpened the known classes but
weakened unknown recall on a proxy (Set B tiles, classes 30–39 treated as unknown, ~430 boxes, never seen by any Set A
model). A4 → B_final3 was the best trade-off once the unknown score was softened (γ 0.25 below). A5 → B_final4 reached
0.414 A-val but scored lower on the proxy.

Set A val (952 tiles) is never trained on; it measures base-class accuracy and forgetting at every stage.

## Pause / resume (control panel)
```bash
python -m anveshak.control                      # panel on http://localhost:8790 (the LOQ starts it at login)
python -m anveshak.train --detach --name A ...  # background run, log in runs/A/train.log
```
**Pause after this epoch** drops `runs/<name>/PAUSE`. Training exits right after the next `last.pt` save, with the
optimizer state intact. **Resume** runs `python -m anveshak.train --resume runs/<name>/weights/last.pt` and continues
from the next epoch. A crashed or rebooted run shows as `stopped` and resumes the same way. Every resume is
recorded in `provenance.json`. **Stop now** (with a confirmation prompt) drops `runs/<name>/STOP`. Training stops within one batch, losing only the epoch in progress.

On the LOQ (native Windows), `powershell -ExecutionPolicy Bypass -File scripts\loq\setup_windows.ps1` sets everything up, including a **desktop shortcut "Anveshak Training"** that opens the panel (starting it if needed), the panel at Windows login, and no sleep on AC. See [docs/LOQ_HANDOFF.md](docs/LOQ_HANDOFF.md).

## Inference, scoring, CSV
```bash
python -m anveshak.infer --weights runs/B_final3/weights/last.pt --images ~/data/round1/test/images \
    --tiles ~/data/round1/test/tiles.csv --classes ~/data/round1/class_list.csv --tta \
    --out runs/submissions/r1_final3_g025.csv --validator ~/data/round1/validate_predictions.py
```
- Post-processing thresholds live in [`configs/postproc.yaml`](configs/postproc.yaml) (tuned on validation only).
- Unknown score = max(s_unknown, s_objectness · (1 − max known score)^γ), γ = 0.25. Kept when ≥ 0.10, at least 48 px,
  at most 10 per tile. The competition's known-vs-unknown conflict rule is applied before writing.
- `--tta` merges four flipped views with the same NMS.
- `tools/oe_score.py` re-implements the published OE metric. `tools/make_csv.py` writes CSVs that pass `validate_predictions.py`.

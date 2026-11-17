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
```bash
python -m anveshak.train --name A --data ~/data/round1/yolo/data_A.yaml --scale m epochs=180 imgsz=1024 batch=4 nbs=64 optimizer=SGD
python -m anveshak.train --name B --data ~/data/round1/yolo/data_B.yaml --from runs/A/weights/last.pt freeze=10 ...
```

## Inference, scoring, CSV
Built in phases P1–P3. See [PLAN.md §7–8](PLAN.md).
`tools/oe_score.py` re-implements the published OE metric. `tools/make_csv.py` writes CSVs that pass `validate_predictions.py`.

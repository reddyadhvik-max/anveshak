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

## Pause / resume (control panel)
```bash
python -m anveshak.control                      # panel on http://localhost:8765 (the LOQ starts it at login)
python -m anveshak.train --detach --name A ...  # background run, log in runs/A/train.log
```
**Pause after this epoch** drops `runs/<name>/PAUSE`. Training exits right after the next `last.pt` save, with the
optimizer state intact. **Resume** runs `python -m anveshak.train --resume runs/<name>/weights/last.pt` and continues
from the next epoch. A crashed or rebooted run shows as `stopped` and resumes the same way. Every resume is
recorded in `provenance.json`. **Stop now** (with a confirmation prompt) drops `runs/<name>/STOP`. Training stops within one batch, losing only the epoch in progress.

On the LOQ (native Windows), `powershell -ExecutionPolicy Bypass -File scripts\loq\setup_windows.ps1` sets everything up, including a **desktop shortcut "Anveshak Training"** that opens the panel (starting it if needed), the panel at Windows login, and no sleep on AC. See [docs/LOQ_HANDOFF.md](docs/LOQ_HANDOFF.md).

## Inference, scoring, CSV
Built in phases P1–P3. See [PLAN.md §7–8](PLAN.md).
`tools/oe_score.py` re-implements the published OE metric. `tools/make_csv.py` writes CSVs that pass `validate_predictions.py`.

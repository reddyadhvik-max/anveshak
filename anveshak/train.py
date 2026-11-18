"""Stage runner for the single model lineage.

  First stage (random init, scale s):
    python -m anveshak.train --name A --data ~/data/round1/yolo/data_A.yaml --scale s epochs=200 imgsz=1024 batch=8
  Every later stage continues from the previous checkpoint (never from a yaml again):
    python -m anveshak.train --name B --data .../data_B.yaml --from runs/A/weights/last.pt epochs=60 freeze=10
  Pause after the current epoch: touch runs/<name>/PAUSE (or the Pause button in `python -m anveshak.control`)
  Resume:                        python -m anveshak.train --resume runs/<name>/weights/last.pt

Extra key=value args go straight to Ultralytics' train(). runs/<name>/provenance.json records where each
stage started, which proves the lineage A -> B -> R2 from one random init.
"""
import argparse
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import torch

from anveshak import patch

CFG = Path(__file__).resolve().parent.parent / "configs" / "yolo11-p2.yaml"


def parse_kv(items):
    from ultralytics.cfg import smart_value  # same parsing as the Ultralytics CLI
    return {k: smart_value(v) for k, v in (it.split("=", 1) for it in items)}


def set_state(run, state):
    """runs/<name>/state.json is what the control panel reads."""
    (run / "state.json").write_text(json.dumps({"state": state, "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))


def pause_after_save(trainer):
    """Called right after last.pt is written each epoch. If runs/<name>/PAUSE exists, exit now.
    Ultralytics' normal stop path runs final_eval, which strips the optimizer from last.pt and makes it
    unresumable. Exiting here leaves last.pt exactly as a crash would, and resume=True continues from it."""
    run = Path(trainer.save_dir)
    flag = run / "PAUSE"
    if flag.exists() and trainer.epoch + 1 < trainer.epochs:  # on the final epoch, let it finish normally
        flag.unlink()
        set_state(run, "paused")
        print(f"\nPaused after epoch {trainer.epoch + 1}/{trainer.epochs}. Resume from {run / 'weights' / 'last.pt'}", flush=True)
        raise SystemExit(0)


def _sigterm(*_):
    raise KeyboardInterrupt  # "Stop now" button; background jobs ignore SIGINT, so the panel sends SIGTERM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--data")
    ap.add_argument("--from", dest="src", help="checkpoint of the previous stage")
    ap.add_argument("--resume", help="runs/<name>/weights/last.pt of a paused or crashed run")
    ap.add_argument("--scale", default="s", help="only used for the first stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("kv", nargs="*")
    a = ap.parse_args()
    if not a.resume and not (a.name and a.data):
        ap.error("--name and --data are required unless --resume is given")
    patch.apply()
    from ultralytics import YOLO

    if a.resume:
        last = Path(a.resume).resolve()
        run = last.parent.parent
        model = YOLO(str(last))
        prov = json.loads((run / "provenance.json").read_text())
        prov.setdefault("resumes", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "from_sha256": patch.weights_sha256(model.model)})
        train_kwargs = {"resume": True}
    else:
        if a.src:
            model = YOLO(a.src)
            prov = {"stage": a.name, "start": "checkpoint", "from": a.src,
                    "from_sha256": patch.weights_sha256(model.model)}
        else:
            torch.manual_seed(a.seed)
            tmp = Path(tempfile.mkdtemp()) / f"yolo11{a.scale}-p2.yaml"  # Ultralytics reads the scale from the filename
            shutil.copy(CFG, tmp)
            model = YOLO(str(tmp))
            prov = {"stage": a.name, "start": "random_init", "scale": a.scale, "seed": a.seed,
                    "init_sha256": patch.weights_sha256(model.model)}
        kv = {"pretrained": False, "seed": a.seed, "deterministic": True,
              "cls": 1.0,  # the objectness slot doubles target_scores_sum; 2x the default cls gain restores the class-loss scale
              "flipud": 0.5, "max_det": 1000, **parse_kv(a.kv)}
        prov["train_args"] = kv
        run = (Path(a.project) / a.name).resolve()
        run.mkdir(parents=True, exist_ok=True)
        train_kwargs = dict(data=str(Path(a.data).expanduser()), project=str(run.parent), name=a.name, exist_ok=True, **kv)

    (run / "provenance.json").write_text(json.dumps(prov, indent=1))
    signal.signal(signal.SIGTERM, _sigterm)
    model.add_callback("on_model_save", pause_after_save)
    set_state(run, "running")
    try:
        model.train(**train_kwargs)
    except SystemExit:
        raise
    except BaseException:
        set_state(run, "stopped")  # crash or Ctrl+C: last.pt is still resumable
        raise
    set_state(run, "finished")


if __name__ == "__main__":
    main()

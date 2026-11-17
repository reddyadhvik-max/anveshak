"""Stage runner for the single model lineage.

  First stage (random init, scale s):
    python -m anveshak.train --name A --data ~/data/round1/yolo/data_A.yaml --scale s epochs=200 imgsz=1024 batch=8
  Every later stage continues from the previous checkpoint (never from a yaml again):
    python -m anveshak.train --name B --data .../data_B.yaml --from runs/A/weights/last.pt epochs=60 freeze=10

Extra key=value args go straight to Ultralytics' train(). runs/<name>/provenance.json records where each
stage started, which proves the lineage A -> B -> R2 from one random init.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from anveshak import patch

CFG = Path(__file__).resolve().parent.parent / "configs" / "yolo11-p2.yaml"


def parse_kv(items):
    from ultralytics.cfg import smart_value  # same parsing as the Ultralytics CLI
    return {k: smart_value(v) for k, v in (it.split("=", 1) for it in items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--from", dest="src", help="checkpoint of the previous stage")
    ap.add_argument("--scale", default="s", help="only used for the first stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("kv", nargs="*")
    a = ap.parse_args()
    patch.apply()
    from ultralytics import YOLO

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
    out = Path(a.project) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    model.train(data=str(Path(a.data).expanduser()), project=str(Path(a.project).resolve()), name=a.name,
                exist_ok=True, **kv)


if __name__ == "__main__":
    main()

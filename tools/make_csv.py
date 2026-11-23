"""Write a submission CSV that always passes validate_predictions.py, then run the validator on it."""
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HEADER = ["image_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"]
MAX_ROWS = 1000


def write_csv(rows, path, tile_sizes):
    """rows: iterable of (tile, label, score, x1, y1, x2, y2). tile_sizes: tile -> (w, h).
    Clips to the tile, rounds, drops degenerate boxes and duplicates, keeps the top 1000 per tile."""
    per = defaultdict(dict)
    for t, l, s, x1, y1, x2, y2 in rows:
        w, h = tile_sizes[t]
        x1, x2 = round(min(max(x1, 0), w), 2), round(min(max(x2, 0), w), 2)
        y1, y2 = round(min(max(y1, 0), h), 2), round(min(max(y2, 0), h), 2)
        s = round(min(max(float(s), 0.0), 1.0), 5)
        if x1 < x2 and y1 < y2:
            key = (l, x1, y1, x2, y2)
            per[t][key] = max(s, per[t].get(key, -1))  # same box+label twice: keep the higher score
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # e.g. runs/submissions/ on a fresh machine
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        n = 0
        for t in sorted(per):
            best = sorted(per[t].items(), key=lambda kv: -kv[1])[:MAX_ROWS]
            for (l, x1, y1, x2, y2), s in best:
                wr.writerow([t, l, f"{s:.5f}", x1, y1, x2, y2])
                n += 1
    return n


def validate(csv_path, validator, tiles_csv, class_csvs):
    r = subprocess.run([sys.executable, validator, csv_path, "--tiles", tiles_csv, "--classes", *class_csvs],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        raise SystemExit(f"validator rejected {csv_path}")

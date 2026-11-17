"""Dataset audit: tiles, boxes, class counts, box sizes. Prints markdown for the phase reports.

  python tools/eda.py --root ~/data/round1
"""
import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    root = ap.parse_args().root.expanduser()
    for ds in ("dataset_A", "dataset_B"):
        boxes, tiles, sizes, per = Counter(), Counter(), [], []
        for f in sorted((root / ds / "labels").glob("*.json")):
            a = json.loads(f.read_text())["annotations"]
            per.append(len(a))
            for c in {x["class_id"] for x in a}:
                tiles[c] += 1
            for x in a:
                boxes[x["class_id"]] += 1
                b = x["bbox"]
                sizes.append(((b[2] - b[0]) * (b[3] - b[1])) ** 0.5)
        q = sorted(sizes)
        n = len(per)
        print(f"### {ds}\n")
        print(f"- tiles: {n} ({sum(p == 0 for p in per)} with no labels)")
        print(f"- boxes: {len(q)}; per tile median {st.median(per)}, p90 {sorted(per)[int(.9 * n)]}, max {max(per)}")
        print(f"- box size (sqrt area, px) p5/p25/p50/p75/p95: {[round(q[int(len(q) * p)]) for p in (.05, .25, .5, .75, .95)]}")
        print(f"- under 16 px: {100 * sum(s < 16 for s in q) / len(q):.1f}%, under 32 px: {100 * sum(s < 32 for s in q) / len(q):.1f}%\n")
        print("| class_id | boxes | tiles |\n|---:|---:|---:|")
        for c in sorted(boxes):
            print(f"| {c} | {boxes[c]} | {tiles[c]} |")
        print()


if __name__ == "__main__":
    main()

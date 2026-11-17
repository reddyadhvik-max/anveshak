import os
from pathlib import Path
import pytest
from make_csv import write_csv, validate

META = Path(os.environ.get("ANVESHAK_DATA", Path(__file__).resolve().parents[2] / "data_meta" / "round1"))


@pytest.mark.skipif(not (META / "validate_predictions.py").exists(), reason="round1 metadata not present")
def test_messy_rows_pass_official_validator(tmp_path):
    import csv
    tiles = {r["tile_id"]: (float(r["width"]), float(r["height"]))
             for r in csv.DictReader(open(META / "test" / "tiles.csv"))}
    t = sorted(tiles)[0]
    rows = [(t, "unknown", 1.7, -5, -5, 30, 30),          # score > 1, box outside tile
            (t, "unknown", 0.4, -5, -5, 30, 30),          # same box -> deduped
            (t, "g53ktwtqo6v", 0.3, 10, 10, 10, 20),      # degenerate -> dropped
            ] + [(t, "g53ktwtqo6v", i / 2000, i % 1000, 0, i % 1000 + 5, 5) for i in range(1500)]  # > 1000 rows
    out = tmp_path / "p.csv"
    n = write_csv(rows, out, tiles)
    assert n == 1000
    validate(str(out), str(META / "validate_predictions.py"), str(META / "test" / "tiles.csv"), [str(META / "class_list.csv")])

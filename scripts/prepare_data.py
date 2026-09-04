"""Unpack SECOM and write the dataset profile the report reads.

    python scripts/prepare_data.py

Extracts ``secom.zip`` into ``data/`` if it is not there yet, then writes
``runs/data_profile.json``: shape, class balance, missingness, constant and
duplicated sensors, and the effect of the fold-internal cleaner. Every dataset
number in the README comes from this file.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from yieldrca.data import load_secom, secom_profile
from yieldrca.preprocess import MissingIndicatorAppender, SensorCleaner
from yieldrca.stability import cluster_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="secom.zip")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/data_profile.json")
    a = ap.parse_args()

    root = Path(a.root)
    if not (root / "secom.data").exists():
        z = Path(a.zip)
        if not z.exists():
            raise SystemExit(f"{z} not found and {root}/secom.data missing")
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        print(f"extracted {z} -> {root}/")

    X, y, names, t = load_secom(a.root, with_time=True)
    prof = secom_profile(X, y)
    prof["time_span_days"] = float((t[-1] - t[0]) / 86400.0)
    prof["time_monotonic"] = bool((np.diff(t) >= 0).all())

    cl = SensorCleaner().fit(X)
    counts = cl.drop_counts()
    prof["cleaner"] = {
        "sensors_in": int(X.shape[1]),
        "sensors_kept": int(len(cl.keep_)),
        "dropped_total": int(X.shape[1] - len(cl.keep_)),
        "dropped_by_reason": counts,
        "exact_duplicate_groups_after_constant_drop": int(len(cl.duplicate_groups_)),
    }
    mi = MissingIndicatorAppender(min_frac=0.01).fit(cl.transform(X))
    prof["cleaner"]["missing_indicator_columns"] = int(len(mi.cols_))

    for th in (0.99, 0.95, 0.90):
        cmap, n_multi = cluster_map(X, thresh=th)
        sizes: dict[int, int] = {}
        for c in cmap.values():
            sizes[c] = sizes.get(c, 0) + 1
        kept = set(int(j) for j in cl.keep_)
        kept_clusters = {cmap[j] for j in kept}
        prof[f"corr_clusters_at_{th}"] = {
            "clusters_over_kept_sensors": int(len(kept_clusters)),
            "kept_sensors": int(len(kept)),
            "largest_cluster": int(max(sizes.values())),
            "sensors_in_a_multi_member_cluster":
                int(sum(n for c, n in sizes.items() if n > 1 and c in kept_clusters)),
        }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(prof, indent=2) + "\n")
    print(json.dumps(prof, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

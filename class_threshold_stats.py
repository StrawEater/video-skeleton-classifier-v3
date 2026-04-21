"""
Report OakInkV2 class/segment counts for a given min-samples threshold.

Usage:
    python dataset_scripts/class_threshold_stats.py --threshold 11
    python dataset_scripts/class_threshold_stats.py --threshold 11 --data-root /data/data3/...
"""

import argparse
import pandas as pd
from pathlib import Path

DEFAULT_ROOT = Path("/data/data3/junibg-ego/Proyectos/skeleton-video-classifier/DATA/OakInkV2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", "-t", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.data_root
    all_segs = pd.read_csv(root / "action_segments.txt", sep="\t")
    counts = all_segs["label_id"].value_counts()
    valid = set(counts[counts >= args.threshold].index)

    kept = all_segs[all_segs["label_id"].isin(valid)]["label_id"].value_counts()

    print(f"Threshold : >= {args.threshold}")
    print(f"Classes   : {len(valid)}")
    print(f"Total segs: {all_segs['label_id'].isin(valid).sum()}")
    print(f"  mean    : {kept.mean():.1f}")
    print(f"  median  : {kept.median():.1f}")
    print(f"  min     : {kept.min()}")
    print(f"  max     : {kept.max()}")
    print()

    for s in ["train", "val", "test"]:
        df = pd.read_csv(root / "label_split" / f"action_{s}.txt", sep="\t")
        kept_n = df["label_id"].isin(valid).sum()
        print(f"  {s:5s}: {kept_n} / {len(df)}")


if __name__ == "__main__":
    main()

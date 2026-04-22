"""
Report OakInkV2 class/segment counts for a given min-samples threshold.

Usage:
    python dataset_scripts/class_threshold_stats.py --threshold 11
    python dataset_scripts/class_threshold_stats.py --threshold 11 --action-only
    python dataset_scripts/class_threshold_stats.py --threshold 11 --data-root /data/data3/...
"""

import argparse
import json
import pandas as pd
from pathlib import Path

DEFAULT_ROOT = Path("/data/data3/junibg-ego/Proyectos/skeleton-video-classifier/DATA/OakInkV2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", "-t", type=int, required=True)
    parser.add_argument("--action-only", "-a", action="store_true",
                        help="Collapse (object, action) pairs to action-only classes")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.data_root
    all_segs = pd.read_csv(root / "action_segments.txt", sep="\t")

    if args.action_only:
        lm = json.loads((root / "label_map.json").read_text())
        id_to_action = {int(k): v["action"] for k, v in lm.items()}
        all_segs["effective"] = all_segs["label_id"].map(id_to_action)
        class_label = "action"
    else:
        all_segs["effective"] = all_segs["label_id"]
        class_label = "(object, action)"

    counts = all_segs["effective"].value_counts()
    valid = set(counts[counts >= args.threshold].index)
    kept_counts = counts[counts >= args.threshold]

    print(f"Mode      : {class_label}")
    print(f"Threshold : >= {args.threshold}")
    print(f"Classes   : {len(valid)}")
    print(f"Total segs: {all_segs['effective'].isin(valid).sum()}")
    print(f"  mean    : {kept_counts.mean():.1f}")
    print(f"  median  : {kept_counts.median():.1f}")
    print(f"  min     : {kept_counts.min()}")
    print(f"  max     : {kept_counts.max()}")
    print()

    for s in ["train", "val", "test"]:
        df = pd.read_csv(root / "label_split" / f"action_{s}.txt", sep="\t")
        if args.action_only:
            df["effective"] = df["label_id"].map(id_to_action)
        else:
            df["effective"] = df["label_id"]
        kept_n = df["effective"].isin(valid).sum()
        print(f"  {s:5s}: {kept_n} / {len(df)}")


if __name__ == "__main__":
    main()

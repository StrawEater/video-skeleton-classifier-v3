"""
Download OakInkV2_hand_full from Hugging Face and reconstruct the dataset:

  <output_dir>/
    scenes/<scene_id>/<orig_frame_id:06d>.jpg
    hand_keypoints/<scene_id>.npy       # float32 (N, 2, 21, 3) — wrist-centered
    wrist_positions/<scene_id>.npy      # float32 (N, 2, 3)     — absolute wrist
    hand_keypoints/<scene_id>_frame_ids.npy
    action_segments.txt
    label_map.json                      # 262 classes (hold removed)
    label_split/
    label_split_trimmed/

Usage:
  python download_oakink2_hf.py [--output-dir DIR]

Requires: huggingface_hub, pyarrow, numpy, tqdm
"""

import argparse
import collections
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID  = "juanito889/OakInkV2_hand_full"
KP_SHAPE = (2, 21, 3)
WP_SHAPE = (2, 3)
DTYPE    = np.float32

METADATA_FILES = ["action_segments.txt", "label_map.json", "README_labels.md"]
METADATA_DIRS  = ["label_split", "label_split_trimmed"]


def reconstruct_shard(parquet_path: Path, output_dir: Path):
    scenes_dir = output_dir / "scenes"
    partial = collections.defaultdict(lambda: {"keypoints": {}, "wrist_positions": {}, "frame_ids": {}})

    df = pq.read_table(parquet_path).to_pandas()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=parquet_path.name, leave=False):
        scene_id      = row["scene_id"]
        frame_idx     = int(row["frame_idx"])
        orig_frame_id = int(row["orig_frame_id"])

        # Write JPEG
        frame_dir = scenes_dir / scene_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{orig_frame_id:06d}.jpg" if orig_frame_id >= 0 else f"{frame_idx:06d}.jpg"
        img_path = frame_dir / fname
        if not img_path.exists():
            img_path.write_bytes(bytes(row["image"]))

        kp_bytes = bytes(row["keypoints"])
        if kp_bytes:
            partial[scene_id]["keypoints"][frame_idx] = (
                np.frombuffer(kp_bytes, dtype=DTYPE).reshape(KP_SHAPE)
            )

        wp_bytes = bytes(row["wrist_position"])
        if wp_bytes:
            partial[scene_id]["wrist_positions"][frame_idx] = (
                np.frombuffer(wp_bytes, dtype=DTYPE).reshape(WP_SHAPE)
            )

        partial[scene_id]["frame_ids"][frame_idx] = orig_frame_id

    return partial


def flush_arrays(partial: dict, kp_dir: Path, wp_dir: Path):
    kp_dir.mkdir(parents=True, exist_ok=True)
    wp_dir.mkdir(parents=True, exist_ok=True)

    for scene_id, data in partial.items():
        if not data["keypoints"]:
            continue
        idxs     = sorted(data["keypoints"].keys())
        kp_stack = np.stack([data["keypoints"][i]       for i in idxs])  # (N, 2, 21, 3)
        fid_arr  = np.array([data["frame_ids"].get(i, -1) for i in idxs], dtype=np.int32)

        kp_path  = kp_dir / f"{scene_id}.npy"
        fid_path = kp_dir / f"{scene_id}_frame_ids.npy"
        if kp_path.exists():
            kp_stack = np.concatenate([np.load(kp_path),  kp_stack], axis=0)
            fid_arr  = np.concatenate([np.load(fid_path), fid_arr],  axis=0)
        np.save(kp_path,  kp_stack)
        np.save(fid_path, fid_arr)

        if data["wrist_positions"]:
            wp_stack = np.stack([data["wrist_positions"][i] for i in idxs])  # (N, 2, 3)
            wp_path  = wp_dir / f"{scene_id}.npy"
            if wp_path.exists():
                wp_stack = np.concatenate([np.load(wp_path), wp_stack], axis=0)
            np.save(wp_path, wp_stack)


def download_metadata(output_dir: Path, repo_id: str, all_files: list[str]):
    for fname in METADATA_FILES:
        if fname in all_files:
            local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname)
            dst = output_dir / fname
            dst.write_bytes(Path(local).read_bytes())
            print(f"  {fname}")
    for dname in METADATA_DIRS:
        split_files = [f for f in all_files if f.startswith(f"{dname}/")]
        for f in split_files:
            local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f)
            dst = output_dir / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(local).read_bytes())
        if split_files:
            print(f"  {dname}/ ({len(split_files)} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "DATA/OakInkV2_jpeg",
                        help="Directory to reconstruct the dataset into")
    parser.add_argument("--repo-id",   type=str,  default=REPO_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repo  : {args.repo_id}")
    print(f"Output: {args.output_dir}")

    all_files   = list(list_repo_files(repo_id=args.repo_id, repo_type="dataset"))
    shard_files = sorted(f for f in all_files if f.endswith(".parquet"))
    print(f"\nFound {len(shard_files)} shards. Downloading metadata...")
    download_metadata(args.output_dir, args.repo_id, all_files)

    kp_dir = args.output_dir / "hand_keypoints"
    wp_dir = args.output_dir / "wrist_positions"

    print(f"\nDownloading and extracting {len(shard_files)} shards...")
    for shard_name in tqdm(shard_files, desc="shards"):
        local_path = hf_hub_download(
            repo_id=args.repo_id, repo_type="dataset",
            filename=shard_name, cache_dir=args.cache_dir,
        )
        partial = reconstruct_shard(Path(local_path), args.output_dir)
        flush_arrays(partial, kp_dir, wp_dir)

    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()

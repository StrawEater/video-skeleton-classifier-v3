"""
Download OakInk2 parquet shards from Hugging Face and reconstruct the original
OakInkV2_jpeg directory structure:

  <output_dir>/
    scenes/<scene_id>/<orig_frame_id:06d>.jpg
    hand_keypoints/<scene_id>.npy          # float32 (N, 2, 21, 3)
    hand_keypoints/<scene_id>_frame_ids.npy  # int32 (N,)
    action_segments.txt
    label_map.json
    label_split/
    ...

Usage:
  python download_oakink2_hf.py --output-dir /data/.../OakInkV2_jpeg

Requires: huggingface_hub, pyarrow, numpy, tqdm
  pip install huggingface_hub pyarrow numpy tqdm
"""

import argparse
import collections
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID   = "juanito889/OakInk2_simplified_parkek"
KP_SHAPE  = (2, 21, 3)
KP_DTYPE  = np.float32
METADATA  = ["action_segments.txt", "label_map.json",
             "incomplete_scenes.txt", "README_labels.md"]


def iter_parquet_rows(parquet_path: Path):
    """Yield dicts for each row in a parquet file without loading it all at once."""
    table = pq.read_table(parquet_path)
    df    = table.to_pandas()
    for _, row in df.iterrows():
        yield row


def reconstruct_shard(parquet_path: Path, output_dir: Path):
    """
    Write images and accumulate keypoints/frame_ids for all scenes in one shard.
    Returns {scene_id: {"keypoints": [...], "frame_ids": [...]}} partial data.
    """
    scenes_dir = output_dir / "scenes"
    partial: dict[str, dict] = collections.defaultdict(lambda: {"keypoints": {}, "frame_ids": {}})

    table = pq.read_table(parquet_path)
    df    = table.to_pandas()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=parquet_path.name, leave=False):
        scene_id      = row["scene_id"]
        frame_idx     = int(row["frame_idx"])
        orig_frame_id = int(row["orig_frame_id"])
        image_bytes   = bytes(row["image"])
        kp_bytes      = bytes(row["keypoints"])

        # Write JPEG
        frame_dir = scenes_dir / scene_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{orig_frame_id:06d}.jpg" if orig_frame_id >= 0 else f"{frame_idx:06d}.jpg"
        img_path = frame_dir / fname
        if not img_path.exists():
            img_path.write_bytes(image_bytes)

        # Accumulate keypoints keyed by frame_idx so order is correct on reassembly
        if kp_bytes:
            kp = np.frombuffer(kp_bytes, dtype=KP_DTYPE).reshape(KP_SHAPE)
            partial[scene_id]["keypoints"][frame_idx] = kp
        partial[scene_id]["frame_ids"][frame_idx] = orig_frame_id

    return partial


def flush_keypoints(partial: dict, kp_dir: Path):
    """Write .npy files for scenes whose data is complete (call after each shard)."""
    kp_dir.mkdir(parents=True, exist_ok=True)
    for scene_id, data in partial.items():
        if not data["keypoints"]:
            continue
        sorted_idxs  = sorted(data["keypoints"].keys())
        kp_stack     = np.stack([data["keypoints"][i] for i in sorted_idxs])   # (N, 2, 21, 3)
        fid_array    = np.array([data["frame_ids"].get(i, -1) for i in sorted_idxs], dtype=np.int32)

        kp_path  = kp_dir / f"{scene_id}.npy"
        fid_path = kp_dir / f"{scene_id}_frame_ids.npy"

        # Append to existing file if shard splits a scene (rare, but handle it)
        if kp_path.exists():
            existing_kp  = np.load(kp_path)
            existing_fid = np.load(fid_path)
            kp_stack  = np.concatenate([existing_kp,  kp_stack],  axis=0)
            fid_array = np.concatenate([existing_fid, fid_array], axis=0)

        np.save(kp_path,  kp_stack)
        np.save(fid_path, fid_array)


def download_metadata(output_dir: Path, repo_files: list[str]):
    """Download flat metadata files and label_split/ directory."""
    for fname in METADATA:
        if fname in repo_files:
            local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=fname)
            dst   = output_dir / fname
            dst.write_bytes(Path(local).read_bytes())
            print(f"  metadata: {fname}")

    split_files = [f for f in repo_files if f.startswith("label_split/")]
    for f in split_files:
        local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f)
        dst   = output_dir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(local).read_bytes())
    if split_files:
        print(f"  metadata: label_split/ ({len(split_files)} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/OakInkV2_jpeg",
                        help="Root directory to reconstruct into")
    parser.add_argument("--repo-id",    type=str, default=REPO_ID)
    parser.add_argument("--cache-dir",  type=Path, default=None,
                        help="HF cache dir for downloaded parquet files (default: ~/.cache/huggingface)")
    args = parser.parse_args()

    print(f"Repo : {args.repo_id}")
    print(f"Output: {args.output_dir}")

    all_files    = list(list_repo_files(repo_id=args.repo_id, repo_type="dataset"))
    shard_files  = sorted(f for f in all_files if f.endswith(".parquet"))
    print(f"Found {len(shard_files)} parquet shards + metadata")

    # Metadata first
    print("\nDownloading metadata...")
    download_metadata(args.output_dir, all_files)

    # Shards
    print(f"\nDownloading and extracting {len(shard_files)} shards...")
    for shard_name in tqdm(shard_files, desc="shards"):
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=shard_name,
            cache_dir=args.cache_dir,
        )
        partial = reconstruct_shard(Path(local_path), args.output_dir)
        flush_keypoints(partial, args.output_dir / "hand_keypoints")

    print(f"\nDone. Reconstructed dataset at: {args.output_dir}")


if __name__ == "__main__":
    main()

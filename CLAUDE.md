# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Conda env: `mamba` — activate before running anything
- Long-running scripts must be launched inside a **tmux** session
- Python path: `/home/juanibg/video-skeleton-classifier/` (server); `/home/juanb/mnt/nikola_home/video-skeleton-classifier/` (local)
- Data root: `/data/data3/junibg-ego/Proyectos/skeleton-video-classifier/DATA/`

## Common Commands

```bash
# Train multimodal model (grid search over 40+ configs)
python training_multimodal.py

# Train single-modality
python training_skeleton.py
python training_video.py

# Evaluate
python evaluation.py           # multimodal
python evaluation_skeleton.py  # skeleton-only
python evaluation_video.py     # video-only

# Dataset preparation (OakInkV2)
python dataset_scripts/create_oakink2_labels.py
python dataset_scripts/extract_oakink2_keypoints.py  # NOT YET WRITTEN

# Quick smoke tests
python test_videomamba.py
python test_mamba.py
```

## Architecture

Three-component pipeline: **VideoMamba** (RGB frames) + **SkeletonMamba** (hand joints) → **MultimodalMambaFusion** (classification).

### SkeletonMamba (`Models/skeleton_mamba.py`)
- Input: `(B, T, 42, 3)` — T frames, 42 joints (21 per hand), xyz coords
- Joint + temporal positional embeddings → configurable Mamba blocks → `(B, 576)` embedding
- Skeleton normalized to `[-1, 1]` range; training uses ±2-frame jitter

### VideoMamba (`VideoMamba/` submodule, OpenGVLab)
- Input: `(B, 3, T, 224, 224)`
- Output: `(B, 576)` — matched to skeleton embedding dim
- Pretrained weights: `videomamba_{size}_{dataset}_f{frames}_res224.pth`

### MultimodalMambaFusion (`Models/multimodal_fusion_mamba.py`)
- Trainable modality embeddings + learnable fusion weights (softmax-normalized)
- Fusion strategies: `'weighted'`, `'context'`, `'new'`, `'average'`
- Orchestrator: `MultimodalActionMamba` — supports missing modalities by zeroing features; single-modality inference via `forward_video_only()` / `forward_skeleton_only()`

### Model sizes
| Size   | skeleton_depth | embed_dim | fusion_depth | ~Params |
|--------|---------------|-----------|--------------|---------|
| tiny   | 12            | 192       | 4            | ~25M    |
| medium | 16            | 192       | 8            | ~35M    |
| large  | 24            | 384       | 12           | ~85M    |

## Datasets

### H2O (integrated)
- Splits: `label_split/action_{train,val,test}.txt` — tab-separated: `id path label_id start_frame end_frame`
- Skeleton: per-frame txt at `{subject}_ego/cam4/hand_pose/{frame:06d}.txt`, 128 floats (flag + 21 joints × 3 per hand)
- Video frames: `{subject}_ego/cam4/rgb256/{frame:06d}.jpg`
- 36 action classes (integer IDs)
- Dataset classes: `H2OSkeletonDataset`, `H2OVideoMambaDataset`, `MultimodalH2ODataset`

### OakInkV2 (in progress)
- Location: `DATA/OakInkV2/`
- Labels derived from `<object, action>` pairs via `program_info` JSONs + affordance/part-tree traversal
- Label outputs: `label_map.json`, `action_segments.txt`, `label_split/`, `label_creation.log`
- Keypoints output: `hand_keypoints/<scene_id>.npy` — shape `(N, 2, 21, 3)`
- `create_oakink2_labels.py` is **done**; `extract_oakink2_keypoints.py` is **not yet written**

## Training Details

- Optimizer: AdamW (`lr=2e-4`, `weight_decay=0.01–0.05`); no weight decay on pos embeddings, layer norms, biases
- Schedule: LinearLR warmup (5%) → CosineAnnealingLR
- Early stopping: LR reduction patience=3, full stop patience=10
- Checkpoints saved to `checkpoints/{ID_PREFIX}/` (best val only)
- Batch size: 8 for `clip_len=8`, 2 for `clip_len=32`

## MANO Keypoint Extraction (Blocked)

MANO forward pass crashes with FPE on old server (CPU instruction incompatibility). Must run on new machine.

- Weights: `mano_v1_2/models/MANO_{RIGHT,LEFT}.pkl`
- Library: `pip install git+https://github.com/lixiny/manotorch.git`
- `chumpy` fix: replace deprecated numpy type aliases in `chumpy/__init__.py`
- `raw_mano` fields: `rh__pose_coeffs (1,16,4)` quaternions → convert to axis-angle `(B,48)` for manotorch; `rh__tsl (1,3)`, `rh__betas (1,10)`

## VideoMamba Build

Requires explicit CUDA builds with env vars:
```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE pip install -e VideoMamba/
```
Needs CUDA compute capability ≥ SM_90.

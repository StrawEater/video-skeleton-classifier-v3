# Project Context — EgoSVis CVPR 2026

## What this project is

Cross-modal egocentric action recognition using Mamba. Two streams: RGB (VideoMamba) + hand skeleton (SkeletonMamba), fused via CLS token strategies. H2O is the primary dataset already integrated.

**Goal of current work:** Add OakInkV2 as a second dataset, converting its annotations into the same format as H2O.

---

## H2O format (target format for all datasets)

- **Labels:** integer ID per action segment representing `<object, action>` pair. No human-readable mapping file — just raw integers.
- **Keypoints:** per-frame txt files, 128 space-separated floats (flag + 21 joints × 3 coords per hand).
- **Splits:** `label_split/action_train.txt`, `action_val.txt`, `action_test.txt` — tab-separated: `id  path  label_id  start_frame  end_frame`

---

## OakInkV2 — what we're building

Location: `/data/data3/junibg-ego/Proyectos/skeleton-video-classifier/DATA/OakInkV2/`

### Key source files
- `program/program_info/*.json` — one JSON per scene, each key is a temporal action segment
- `object_affordance/object_affordance.json` — maps object ID → affordances
- `object_affordance/object_part_tree.json` — maps parent → [children] (traverse UPWARD to find root object)
- `object_repair/obj_desc.json` — maps object ID → human-readable name
- `anno_preview/*.pkl` — per-scene mocap data including `raw_mano` (hand pose params, NOT joint positions)

### program_info key format (3 variants)
```
((lh_start, lh_end), None)              # left hand only  → use obj_list_lh
(None, (rh_start, rh_end))              # right hand only → use obj_list_rh
((lh_start, lh_end), (rh_start, rh_end))# both hands     → use obj_list, interval = min/max
```
Always use `primitive` (not primitive_lh/rh) for the action.

### Label derivation
`<object, action>` pair where:
- **action** = `primitive` field
- **object** = root parent name (traverse `object_part_tree` upward, lookup name in `obj_desc`)
- Object selection: pick from active hand's obj_list the one whose affordance matches the primitive. Fallback to full obj_list. Log fallbacks to `label_creation.log`.

### Outputs being created
- `label_map.json` — int ID → {object, action}
- `action_segments.txt` — all segments (tab-sep)
- `label_split/action_{train,val,test}.txt` — 70/15/15 scene-level stratified split
- `hand_keypoints/<scene_id>.npy` — shape (N, 2, 21, 3): frames × [right, left] × joints × xyz
- `hand_keypoints/<scene_id>_frame_ids.npy` — mocap frame ID alignment
- `label_creation.log` — fallback events for manual review

---

## Scripts

- `dataset_scripts/create_oakink2_labels.py` — label mapping + split (DONE, ready to run)
- `dataset_scripts/extract_oakink2_keypoints.py` — keypoint extraction (NOT YET WRITTEN)

---

## MANO keypoint extraction — BLOCKED on old server

The MANO forward pass crashes with FPE on the old server (CPU instruction issue). Need to run on the new computer.

- MANO weights: `/data/data3/junibg-ego/Proyectos/skeleton-video-classifier/mano_v1_2/models/MANO_RIGHT.pkl` and `MANO_LEFT.pkl`
- `manotorch` can be installed: `pip install git+https://github.com/lixiny/manotorch.git`
- `chumpy` dependency needs numpy patch: in `chumpy/__init__.py` replace `from numpy import bool, int, float, complex, object, unicode, str, nan, inf` with the numpy 1.24+ compatible version
- raw_mano structure per frame: `rh__pose_coeffs (1,16,4)`, `rh__tsl (1,3)`, `rh__betas (1,10)` (same for lh__)
- pose_coeffs are quaternions (16 joints × 4), need conversion to axis-angle (×3) for manotorch: shape (B, 48)

---

## General project info

- Code repo: `/home/juanibg/video-skeleton-classifier/`
- Data: `/data/data3/junibg-ego/Proyectos/skeleton-video-classifier/DATA/`
- MANO: `/data/data3/junibg-ego/Proyectos/skeleton-video-classifier/mano_v1_2/`
- Python env: `mamba` conda env
- Long-running scripts → run in tmux

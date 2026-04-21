# Config Reference

All training is driven by YAML configs passed to `train.py`.

## Inheritance

Every config can declare `base: <path>` to inherit defaults. Fields in the child override the parent.
`base.yaml` is the root defaults file — all dataset-specific configs extend it.

## Checkpoint Subfolder Naming

The subfolder under `checkpoint_dir` is exactly `experiment.name`.

For sweep runs, the trainer appends `__<label1>__<label2>...` for each grid axis, in order:

```
checkpoints/
  oakink2_skeleton_tiny/               # single run
  oakink2_skeleton_tiny__100__tiny/    # sweep: max_train_samples=100, size=tiny
  oakink2_multimodal_tiny__fd4__weighted/
```

Checkpoint files inside follow: `epoch{NNN}_train{top1:.2f}_val{top1:.2f}.pth`.
Only the best validation checkpoint is kept; older ones are deleted automatically.

---

## Field Reference

### `experiment`

| Field | Description |
|---|---|
| `name` | Experiment identifier. Becomes the checkpoint subfolder name. |
| `modality` | `skeleton` \| `video` \| `multimodal` |

---

### `dataset`

| Field | Default | Description |
|---|---|---|
| `name` | — | `h2o` or `oakink2` |
| `train_split` | — | Path to split file (tab-separated: `id path label start end`) |
| `val_split` | — | |
| `test_split` | — | |
| `num_classes` | — | 36 (H2O) or 306 (OakInkV2) |
| `clip_len` | — | Number of frames per clip (`8` or `32`) |
| `num_joints` | 21 | Joints per hand; model input = `num_joints × num_hands = 42` |
| `num_hands` | 2 | |
| `normalize_skeleton` | `true` | Normalize joint coords to `[-1, 1]` |
| `with_jitter` | `true` | ±2-frame temporal jitter during training |
| `max_train_samples` | `null` | Cap training set size; `null` = use all |
| `skeleton_root` | — | H2O only: root directory for per-frame `.txt` keypoint files |
| `skeleton_format` | — | H2O only: `txt` |
| `keypoints_root` | — | OakInkV2 only: directory of `<scene_id>.npy` files, shape `(N, 2, 21, 3)` |
| `frames_root` | — | Video only: root directory for RGB frame images |

---

### `model`

| Field | Default | Description |
|---|---|---|
| `size` | — | `tiny` \| `small` \| `medium` — controls `skeleton_depth`, `embed_dim`, `fusion_depth` |
| `pretrained` | `false` | Load pretrained backbone weights |
| `pretrained_path` | `null` | Explicit `.pth` path; `null` = auto-resolve from `training.pretrained_dir` |
| `video_pretrained` | `false` | Multimodal only |
| `skeleton_pretrained` | `false` | Multimodal only |
| `video_pretrained_path` | `null` | Multimodal only |
| `skeleton_pretrained_path` | `null` | Multimodal only |
| `fusion_depth` | — | Multimodal only: Mamba blocks in the fusion head |
| `fusion_strategy` | — | Multimodal only: `weighted` \| `context` \| `new` \| `average` |

Model size table:

| Size | `skeleton_depth` | `embed_dim` | `fusion_depth` | ~Params |
|---|---|---|---|---|
| tiny | 12 | 192 | 4 | ~25M |
| medium | 16 | 192 | 8 | ~35M |
| large | 24 | 384 | 12 | ~85M |

---

### `training`

| Field | Default | Description |
|---|---|---|
| `checkpoint_dir` | `checkpoints` | Root directory for all checkpoints |
| `pretrained_dir` | `~/mnt/nikola_data/.../PRETRAINED` | Directory synced to cloud; auto-resolve finds `.pth` files here |
| `batch_size` | `null` | `null` = auto: 16 for skeleton/video 8f, 8 for multimodal 8f, 4 for 32f |
| `num_workers` | 8 | DataLoader workers |
| `total_epochs` | 200 | |
| `learning_rate` | `2.0e-4` | AdamW base LR |
| `weight_decay` | 0.05 | Not applied to pos embeddings, layer norms, or biases |
| `warmup_fraction` | 0.05 | Fraction of `total_epochs` used for linear warmup |
| `grad_clip` | 1.0 | Gradient norm clip |
| `lr_patience` | 5 | Epochs without val improvement before LR reduction |
| `lr_factor` | 0.1 | Multiplier applied on LR reduction |
| `stop_patience` | 10 | Epochs without val improvement before early stop |
| `skip_if_exists` | `true` | Skip experiment if checkpoint subfolder already exists (grid search resumption) |

---

## Sweep Configs

Sweep files live under `configs/sweeps/` and produce a grid of experiments from a base config.

```yaml
sweep:
  base: configs/oakink2_skeleton.yaml
  grid:
    dataset.max_train_samples:
      100:  100
      full: null        # label: value
    model.size:
      tiny:   tiny
      medium: medium
```

Grid values can be a **dict** `{label: value}` (readable names) or a **list** (labels default to `str(value)`).
The experiment name for each combination is `<base_name>__<label1>__<label2>...` in grid-key order.

Available sweeps:

| File | Grid | Experiments |
|---|---|---|
| `sweeps/datasize_skeleton.yaml` | `max_train_samples` × `model.size` | 4 × 3 = 12 |
| `sweeps/model_complexity.yaml` | `model.size` × `fusion_depth` × `fusion_strategy` | 3 × 2 × 4 = 24 |

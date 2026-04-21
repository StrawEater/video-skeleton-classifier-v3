# Multi-Modal Hand Skeleton and Video Action Recognition with Mamba

This repository contains a comprehensive implementation of multi-modal action recognition combining:
1. **VideoMamba** - Video-based action recognition using Mamba blocks
2. **SkeletonMamba** - Hand skeleton-based action recognition using Mamba blocks
3. **MultimodalMambaFusion** - Fusion module that combines both modalities with trainable embeddings

## Architecture Overview

### 1. SkeletonMamba Encoder

The `SkeletonMamba` model processes hand skeleton data with variable depth Mamba blocks:

```
Input: (B, T, num_joints, joint_dim)
  ↓
Joint Embedding: Projects joint coordinates to embedding space
  ↓
Add Spatial Positional Embeddings (position in skeleton)
  ↓
Add Temporal Positional Embeddings (across frames)
  ↓
Mamba Blocks (configurable depth)
  ↓
LayerNorm
  ↓
Output: (B, embed_dim)
```

**Key Features:**
- Flexible number of joints (default: 21 for MediaPipe hand skeleton)
- Supports 2D or 3D coordinates
- Positional and temporal embeddings similar to VideoMamba
- Variable depth (small: 12, medium: 16, large: 24 blocks)
- RMSNorm and fused add-norm options for efficiency

### 2. MultimodalMambaFusion Head

The fusion module combines skeleton and video embeddings:

```
Video Features (B, embed_dim)     Skeleton Features (B, embed_dim)
  ↓                                    ↓
Add Video Modality Embedding      Add Skeleton Modality Embedding
  ↓                                    ↓
  └────────────────┬────────────────┘
                   ↓
           Fusion Strategy:
      - Sum: Weighted sum of modalities
      - Concat: Concatenate and project
      - Attention: Multi-head attention fusion
                   ↓
            Mamba Fusion Blocks
                   ↓
              LayerNorm
                   ↓
         Classification Head
                   ↓
         Output: (B, num_classes)
```

**Fusion Features:**
- **Trainable Modality Embeddings**: Distinguish between skeleton and video data
- **Learnable Fusion Weights**: Control the contribution of each modality
- **Multiple Fusion Strategies**: Sum, concatenation, or attention-based
- **Configurable Fusion Depth**: Control fusion complexity (small: 4, medium: 8, large: 12 blocks)

### 3. MultimodalActionMamba

The unified model that orchestrates both encoders and the fusion head:

```
Video Input (B, C, T, H, W)       Skeleton Input (B, T, J, D)
  ↓                                    ↓
VideoMamba Encoder              SkeletonMamba Encoder
  ↓                                    ↓
Video Features (B, 576)          Skeleton Features (B, 576)
  └────────────────┬────────────────┘
                   ↓
        MultimodalMambaFusion
                   ↓
              Logits (B, num_classes)
```

**Capabilities:**
- **Multimodal Training**: Train with both modalities simultaneously
- **Single-Modality Inference**: Test with video-only or skeleton-only (`forward_video_only()`, `forward_skeleton_only()`)
- **Feature Extraction**: Extract intermediate representations without classification (`extract_video_features()`, `extract_skeleton_features()`)
- **Flexible Architecture**: Adjust skeleton depth, embedding dimensions, and fusion strategy

## Models and Factory Functions

### Video-Only Models
```python
from Models.multimodal_action_mamba import VisionActionMamba

model = VisionActionMamba(
    pretrained=True,
    num_classes=36,
    num_frames=8,
    pretrained_path="videomamba_m16_breakfast_mask_ft_f64_res224.pth"
)
```

### Skeleton-Only Models
```python
from Models.skeleton_mamba import (
    SkeletonActionMamba,
    skeleton_mamba_small,
    skeleton_mamba_medium,
    skeleton_mamba_large
)

# Using factory function
model = skeleton_mamba_medium(
    num_joints=21,
    num_frames=8,
    num_classes=36,
    embed_dim=192
)

# Or direct class
model = SkeletonActionMamba(
    num_joints=21,
    joint_dim=3,
    depth=16,
    embed_dim=192,
    num_classes=36,
    num_frames=8
)
```

### Multi-Modal Models
```python
from Models.multimodal_action_mamba import (
    MultimodalActionMamba,
    create_multimodal_mamba_small,
    create_multimodal_mamba_medium,
    create_multimodal_mamba_large
)

# Using factory function
model = create_multimodal_mamba_medium(
    num_classes=36,
    num_frames=8
)

# Or direct class with custom configuration
model = MultimodalActionMamba(
    num_classes=36,
    num_frames=8,
    num_joints=21,
    joint_dim=3,
    video_pretrained=True,
    skeleton_depth=16,
    skeleton_embed_dim=192,
    fusion_depth=8,
    fusion_strategy='sum'
)
```

## Data Loading

### Skeleton Dataset Formats

The `H2OSkeletonDataset` supports multiple skeleton data formats:

1. **JSON Format** - Individual JSON files per frame
```
{skeleton_root}/subject1_ego/h1/
├── 000001.json  # {"landmarks": [[x, y, z], ...]}
├── 000002.json
└── ...
```

2. **NPZ Format** - Batch format with all frames
```
{skeleton_root}/subject1_ego/h1/
└── skeleton.npz  # Contains all frame data
```

3. **Pickle Format** - Python pickle serialization
```
{skeleton_root}/subject1_ego/h1/
└── skeleton.pkl  # Contains skeleton data as dict or list
```

### Usage Example

```python
from Datasets.skeleton_dataset import H2OSkeletonDataset, MultimodalH2ODataset
from torch.utils.data import DataLoader

# Single modality: skeleton only
skeleton_dataset = H2OSkeletonDataset(
    csv_path="Data/H2O/label_split/action_train.txt",
    skeleton_root="Data/H2O",
    clip_len=8,
    num_joints=21,
    skeleton_format='json',
    normalize_skeleton=True,
    training=True
)

skeleton_loader = DataLoader(skeleton_dataset, batch_size=8, shuffle=True)

# Get a batch
skeleton, label = next(iter(skeleton_loader))
# skeleton shape: (B, clip_len, num_joints, joint_dim)
# label shape: (B,)

# Multi-modal: video + skeleton
multimodal_dataset = MultimodalH2ODataset(
    csv_path="Data/H2O/label_split/action_train.txt",
    frames_root="Data/H2O",
    skeleton_root="Data/H2O",
    clip_len=8,
    num_joints=21,
    skeleton_format='json'
)

multimodal_loader = DataLoader(multimodal_dataset, batch_size=8, shuffle=True)

# Get a batch
video_frames, skeleton, label = next(iter(multimodal_loader))
# video_frames shape: (B, C, T, H, W)
# skeleton shape: (B, T, num_joints, joint_dim)
# label shape: (B,)
```

## Training

Use the provided training script `train_multimodal.py` to train different models:

### Train Video-Only Model
```bash
python train_multimodal.py \
    --mode video \
    --train-csv Data/H2O/label_split/action_train.txt \
    --frames-root Data/H2O \
    --skeleton-root Data/H2O \
    --num-classes 36 \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --video-pretrained
```

### Train Skeleton-Only Model
```bash
python train_multimodal.py \
    --mode skeleton \
    --train-csv Data/H2O/label_split/action_train.txt \
    --frames-root Data/H2O \
    --skeleton-root Data/H2O \
    --num-classes 36 \
    --skeleton-depth 16 \
    --epochs 50 \
    --batch-size 8
```

### Train Multi-Modal Model
```bash
python train_multimodal.py \
    --mode multimodal \
    --model-size medium \
    --train-csv Data/H2O/label_split/action_train.txt \
    --frames-root Data/H2O \
    --skeleton-root Data/H2O \
    --num-classes 36 \
    --skeleton-depth 16 \
    --fusion-depth 8 \
    --fusion-strategy sum \
    --epochs 50 \
    --batch-size 8 \
    --video-pretrained \
    --normalize-skeleton
```

## Inference and Feature Extraction

### Basic Inference
```python
import torch
from Models.multimodal_action_mamba import MultimodalActionMamba

model = MultimodalActionMamba(num_classes=36, num_frames=8)
model.eval()

# Load model checkpoint
# model.load_state_dict(torch.load('checkpoint.pth'))

# Forward pass
with torch.no_grad():
    video = torch.randn(1, 3, 8, 224, 224)
    skeleton = torch.randn(1, 8, 21, 3)
    
    logits = model(video, skeleton)  # (1, 36)
```

### Single-Modality Inference
```python
# Video only (skeleton features will be zero)
logits_video_only = model.forward_video_only(video)

# Skeleton only (video features will be zero)
logits_skeleton_only = model.forward_skeleton_only(skeleton)
```

### Feature Extraction
```python
# Extract intermediate representations
video_features = model.extract_video_features(video)  # (1, 576)
skeleton_features = model.extract_skeleton_features(skeleton)  # (1, 576)

# Get fused features before classification
fused_features = model.fusion_head.forward_fusion(video_features, skeleton_features)  # (1, 576)
```

## Configuration Details

### Skeleton Encoder Configurations

| Model | Depth | Embed Dim | Parameters | Speed |
|-------|-------|-----------|-----------|-------|
| Small | 12 | 192 | ~25M | Fast |
| Medium | 16 | 192 | ~35M | Balanced |
| Large | 24 | 384 | ~85M | Slower |

### Fusion Head Configurations

| Model | Depth | Fusion Strategy | Parameters |
|-------|-------|-----------------|-----------|
| Small | 4 | Sum | ~50K |
| Medium | 8 | Sum | ~120K |
| Large | 12 | Sum | ~200K |

### Joint Coordinates

- **2D Skeletons**: (x, y) coordinates - typically from video analysis
- **3D Skeletons**: (x, y, z) coordinates - from depth sensors or 3D pose estimation
- **With Confidence**: Can append confidence scores as additional dimensions

## Model Sizes

### Small Configuration
```python
MultimodalActionMamba(
    skeleton_depth=12,
    skeleton_embed_dim=192,
    fusion_depth=4,
)
```

### Medium Configuration (Recommended)
```python
MultimodalActionMamba(
    skeleton_depth=16,
    skeleton_embed_dim=192,
    fusion_depth=8,
)
```

### Large Configuration
```python
MultimodalActionMamba(
    skeleton_depth=24,
    skeleton_embed_dim=384,
    fusion_depth=12,
)
```

## Normalization and Preprocessing

### Skeleton Normalization
The dataset loader can automatically normalize skeleton coordinates:

```python
dataset = H2OSkeletonDataset(
    ...,
    normalize_skeleton=True  # Scales coordinates to [-1, 1] range
)
```

Normalization is applied per-sample, independently for each axis:
```
normalized = 2 * (x - min) / (max - min) - 1
```

### Data Padding
When action clips are shorter than the requested clip length, padding modes are available:

- `repeat`: Repeat the last frame (default)
- `zero`: Zero padding
- `edge`: Repeat edge values

```python
dataset = H2OSkeletonDataset(
    ...,
    pad_mode='repeat'
)
```

## Advanced Features

### Custom Fusion Strategies

The fusion head supports multiple strategies:

```python
# Sum-based fusion (efficient, default)
model = MultimodalActionMamba(..., fusion_strategy='sum')

# Concatenation with projection
model = MultimodalActionMamba(..., fusion_strategy='concat')

# Attention-based fusion (more expressive)
model = MultimodalActionMamba(..., fusion_strategy='attention')
```

### Checkpoint and Inference

Allocate inference cache for faster generation:

```python
model.video_encoder.allocate_inference_cache(batch_size=1, max_seqlen=1000)
model.skeleton_encoder.allocate_inference_cache(batch_size=1, max_seqlen=1000)
model.fusion_head.allocate_inference_cache(batch_size=1, max_seqlen=1000)
```

### No Weight Decay Parameters

Some parameters should not have weight decay applied:

```python
# This is handled automatically in the models
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    no_decay = any(nd in name for nd in model.video_encoder.no_weight_decay())
    no_decay = no_decay or any(nd in name for nd in model.skeleton_encoder.no_weight_decay())
    no_decay = no_decay or any(nd in name for nd in model.fusion_head.no_weight_decay())
    
    if no_decay:
        no_decay_params.append(param)
    else:
        decay_params.append(param)
```

## Implementation Notes

1. **Modality Embeddings**: The fusion head learns separate embeddings for video and skeleton modalities, allowing the model to weight their contributions.

2. **Shared Embedding Dimension**: Both skeleton and video encoders use the same embedding dimension (576) to ensure compatibility in the fusion layer.

3. **Positional Embeddings**: 
   - Video: Spatial (patch position) + Temporal (frame position)
   - Skeleton: Spatial (joint position in skeleton) + Temporal (frame position)

4. **Flexible Input**: The model can handle missing modalities by using zero vectors during inference.

5. **Trainable Weights**: Fusion weights are normalized using softmax to ensure stable contribution ratios.

## Citation

If you use this implementation, please cite:

```bibtex
@article{videomamba2024,
  title={VideoMamba: State Space Model for Efficient Video Understanding},
  author={Li et al.},
  year={2024}
}

@article{mamba2023,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu and Dao},
  year={2023}
}
```

## License

This implementation is provided as-is for research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

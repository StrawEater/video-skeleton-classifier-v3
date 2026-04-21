"""
QUICKSTART GUIDE: Multi-Modal Hand Skeleton + Video Action Recognition

This guide shows the simplest way to get started with the multi-modal models.
"""

# ============================================================================
# 1. INSTALLATION & IMPORTS
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the multi-modal model
from Models.multimodal_action_mamba import (
    MultimodalActionMamba,
    create_multimodal_mamba_medium,
    VisionActionMamba,
    SkeletonActionMamba,
)

# Import datasets
from Datasets.skeleton_dataset import (
    H2OSkeletonDataset,
    MultimodalH2ODataset,
)
from Datasets.h20_dataset import H2OVideoMambaDataset


# ============================================================================
# 2. QUICK START: Create and Use Models
# ============================================================================

# ----- Option A: Multi-Modal Model (RECOMMENDED) -----
print("Creating multi-modal model...")
model = create_multimodal_mamba_medium(num_classes=36, num_frames=8)

# Create dummy data
video = torch.randn(2, 3, 8, 224, 224)  # (batch, channels, frames, height, width)
skeleton = torch.randn(2, 8, 21, 3)      # (batch, frames, joints, coordinates)

# Forward pass
with torch.no_grad():
    logits = model(video, skeleton)
    print(f"Output shape: {logits.shape}")  # (2, 36)


# ----- Option B: Video-Only Model -----
print("\nCreating video-only model...")
video_model = VisionActionMamba(
    pretrained=True,
    num_classes=36,
    num_frames=8,
    pretrained_path="videomamba_m16_breakfast_mask_ft_f64_res224.pth"
)
with torch.no_grad():
    logits = video_model(video)


# ----- Option C: Skeleton-Only Model -----
print("\nCreating skeleton-only model...")
skeleton_model = SkeletonActionMamba(
    num_joints=21,
    joint_dim=3,
    depth=16,
    embed_dim=192,
    num_classes=36,
    num_frames=8
)
with torch.no_grad():
    logits = skeleton_model(skeleton)


# ============================================================================
# 3. LOADING DATA
# ============================================================================

# Load skeleton dataset
skeleton_dataset = H2OSkeletonDataset(
    csv_path="Data/H2O/label_split/action_train.txt",
    skeleton_root="Data/H2O",
    clip_len=8,
    num_joints=21,
    skeleton_format='json',  # or 'npz', 'pkl'
    normalize_skeleton=True,
    training=True
)

skeleton_loader = DataLoader(skeleton_dataset, batch_size=8, shuffle=True)

# Load multi-modal dataset (video + skeleton)
multimodal_dataset = MultimodalH2ODataset(
    csv_path="Data/H2O/label_split/action_train.txt",
    frames_root="Data/H2O",
    skeleton_root="Data/H2O",
    clip_len=8,
    num_joints=21,
    skeleton_format='json'
)

multimodal_loader = DataLoader(multimodal_dataset, batch_size=8, shuffle=True)

# Get a sample batch
video_batch, skeleton_batch, labels_batch = next(iter(multimodal_loader))
print(f"\nBatch shapes:")
print(f"  Video: {video_batch.shape}")
print(f"  Skeleton: {skeleton_batch.shape}")
print(f"  Labels: {labels_batch.shape}")


# ============================================================================
# 4. TRAINING SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_multimodal_mamba_medium(num_classes=36).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    for batch_idx, (video, skeleton, labels) in enumerate(multimodal_loader):
        video = video.to(device)
        skeleton = skeleton.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(video, skeleton)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")


# ============================================================================
# 5. INFERENCE
# ============================================================================

model.eval()

with torch.no_grad():
    # Full multimodal inference
    logits = model(video, skeleton)
    predictions = torch.argmax(logits, dim=1)
    confidences = torch.softmax(logits, dim=1).max(dim=1)[0]
    
    print(f"\nFull Multimodal Inference:")
    print(f"  Predictions: {predictions}")
    print(f"  Confidences: {confidences}")
    
    # Video-only inference (if skeleton is not available)
    logits_video_only = model.forward_video_only(video)
    print(f"\nVideo-Only Inference:")
    print(f"  Predictions: {torch.argmax(logits_video_only, dim=1)}")
    
    # Skeleton-only inference (if video is not available)
    logits_skeleton_only = model.forward_skeleton_only(skeleton)
    print(f"\nSkeleton-Only Inference:")
    print(f"  Predictions: {torch.argmax(logits_skeleton_only, dim=1)}")


# ============================================================================
# 6. FEATURE EXTRACTION
# ============================================================================

# Extract features for downstream tasks
with torch.no_grad():
    video_features = model.extract_video_features(video)          # (B, 576)
    skeleton_features = model.extract_skeleton_features(skeleton)  # (B, 576)
    fused_features = model.fusion_head.forward_fusion(
        video_features, 
        skeleton_features
    )  # (B, 576)

print(f"\nFeature Extraction:")
print(f"  Video features: {video_features.shape}")
print(f"  Skeleton features: {skeleton_features.shape}")
print(f"  Fused features: {fused_features.shape}")


# ============================================================================
# 7. SAVING AND LOADING MODELS
# ============================================================================

# Save checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, 'checkpoint.pth')
print("\nModel saved to 'checkpoint.pth'")

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
_, epoch = checkpoint['epoch']
print(f"Model loaded from checkpoint (epoch {epoch})")


# ============================================================================
# 8. CUSTOM CONFIGURATION
# ============================================================================

# Create model with custom configuration
custom_model = MultimodalActionMamba(
    num_classes=36,
    num_frames=8,
    num_joints=21,
    joint_dim=3,
    # Video parameters
    video_pretrained=True,
    video_pretrained_path="videomamba_m16_breakfast_mask_ft_f64_res224.pth",
    # Skeleton parameters
    skeleton_depth=16,          # Depth of skeleton encoder
    skeleton_embed_dim=192,     # Embedding dimension for skeleton
    # Fusion parameters
    fusion_depth=8,             # Depth of fusion head
    fusion_strategy='sum',      # Options: 'sum', 'concat', 'attention'
)

print(f"\nCustom Multi-Modal Model created with:")
print(f"  Skeleton depth: 16")
print(f"  Skeleton embed dim: 192")
print(f"  Fusion depth: 8")
print(f"  Fusion strategy: sum")


# ============================================================================
# HELPFUL NOTES
# ============================================================================

"""
KEY POINTS:

1. Model Variants:
   - create_multimodal_mamba_small(): Fast, ~25M params
   - create_multimodal_mamba_medium(): Balanced, ~35M params (recommended)
   - create_multimodal_mamba_large(): Powerful, ~85M params

2. Input Shapes:
   - Video: (B, C=3, T, H=224, W=224)
   - Skeleton: (B, T, J=21, D=3)
   - Both must have same batch size B and temporal length T

3. Skeleton Data Formats:
   - JSON: Individual files per frame (flexible, slower loading)
   - NPZ: Batch format (efficient, memory-friendly)
   - PKL: Python pickle (flexible format)

4. Single-Modality Support:
   - forward_video_only(): Video features are used, skeleton is zeroed
   - forward_skeleton_only(): Skeleton features are used, video is zeroed
   - forward(): Both modalities combined (recommended)

5. Feature Extraction:
   - extract_video_features(): (B, 576)
   - extract_skeleton_features(): (B, 576)
   - Can use for clustering, similarity search, etc.

6. Training Tips:
   - Use AdamW optimizer with lr=1e-4 to 1e-5
   - Batch size 8-16 recommended
   - Normalize skeleton coordinates for better convergence
   - Use warmup for first few epochs

7. Fusion Strategies:
   - 'sum': Weighted combination (fastest, ~0.1ms overhead)
   - 'concat': Project concatenated features (balanced)
   - 'attention': Multi-head attention fusion (more expressive)

For more details, see MULTIMODAL_README.md
"""

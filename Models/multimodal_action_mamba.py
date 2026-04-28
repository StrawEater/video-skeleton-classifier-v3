import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
from VideoMamba.videomamba.video_sm.models.videomamba import (
    videomamba_middle, videomamba_small, videomamba_tiny,
    load_state_dict as vm_load_state_dict,
)
from .skeleton_mamba import skeleton_mamba_tiny, skeleton_mamba_small, skeleton_mamba_medium
from .multimodal_fusion_mamba import MultimodalMambaFusion


class VisionActionMamba(nn.Module):
    """Single-modality model: Video-only action recognition using VideoMamba"""
    def __init__(self, pretrained=False, num_classes=36, num_frames=8, pretrained_path=None):
        super().__init__()
        self.backbone = videomamba_middle(pretrained=pretrained, num_frames=num_frames, pretrained_path=pretrained_path)
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


class SkeletonActionMamba(nn.Module):
    """Single-modality model: Skeleton-only action recognition using SkeletonMamba"""
    def __init__(
        self, 
        num_joints=21,
        joint_dim=3,
        depth=16,
        embed_dim=192,
        num_classes=36,
        num_frames=8,
        pretrained=False,
    ):
        super().__init__()
        self.skeleton_encoder = skeleton_mamba_medium(
            num_joints=num_joints,
            num_frames=num_frames,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            return_features_only=False,
        )

    def forward(self, skeleton_data):
        """
        Args:
            skeleton_data: (B, T, num_joints, joint_dim) skeleton coordinates
        Returns:
            logits: (B, num_classes) action predictions
        """
        logits = self.skeleton_encoder(skeleton_data)
        return logits


class MultimodalActionMamba(nn.Module):
    """
    Unified multi-modal action recognition model that combines:
    1. Video encoder (VideoMamba)
    2. Skeleton encoder (SkeletonMamba)
    3. Fusion head (MultimodalMambaFusion)
    
    The architecture extracts independent representations from video and skeleton,
    then fuses them using a Mamba-based fusion head with trainable modality embeddings.
    """
    def __init__(
        self,
        num_classes=36,
        num_frames=8,
        num_joints=42,  # 21 joints per hand * 2 hands
        joint_dim=3,
        # Model size parameters
        video_model_size='medium',
        skeleton_model_size='medium',
        # Pretraining options
        video_pretrained=True,
        video_pretrained_path=None,
        skeleton_pretrained=True,
        skeleton_pretrained_path=None,
        # Fusion head parameters
        fusion_model_depth=8,
        fusion_strategy='weighted',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.joint_dim = joint_dim

        _vid_factory = {'tiny': videomamba_tiny, 'small': videomamba_small, 'medium': videomamba_middle}
        self.video_encoder = _vid_factory.get(video_model_size, videomamba_middle)(
            pretrained=False,
            num_frames=num_frames,
            num_classes=-1,
        )
        if video_pretrained and video_pretrained_path:
            state = torch.load(video_pretrained_path, map_location='cpu')
            vm_load_state_dict(self.video_encoder, state, center=True)
            print(f"Loaded video weights: {video_pretrained_path}")

        self.video_encoder.head = nn.Identity()  # Remove classification head

        self.video_embed_dim = self.video_encoder.embed_dim # Use the embed_dim from the video encoder for consistency in fusion

        if skeleton_model_size == 'tiny':
            self.skeleton_encoder = skeleton_mamba_tiny(
                num_joints=num_joints,
                num_frames=num_frames,
                num_classes=-1,  # Return features only
                embed_dim=self.video_embed_dim,  # Match video embed_dim
                pretrained=skeleton_pretrained,
                pretrained_path=skeleton_pretrained_path,
                return_features_only=True,
            )
        
        elif skeleton_model_size == 'small':
            self.skeleton_encoder = skeleton_mamba_small(
                num_joints=num_joints,
                num_frames=num_frames,
                num_classes=-1,  
                embed_dim=self.video_embed_dim,  
                pretrained=skeleton_pretrained,
                pretrained_path=skeleton_pretrained_path,
                return_features_only=True,
            )
        else:
            self.skeleton_encoder = skeleton_mamba_medium(
                num_joints=num_joints,
                num_frames=num_frames,
                num_classes=-1,  
                embed_dim=self.video_embed_dim,  
                pretrained=skeleton_pretrained,
                pretrained_path=skeleton_pretrained_path,
                return_features_only=True,
            )

        self.fusion_head = MultimodalMambaFusion(
            embed_dim = self.video_embed_dim,
            fusion_depth=fusion_model_depth,
            num_classes=num_classes,
            fusion_strategy=fusion_strategy,
        )

    def forward(self, video_input, skeleton_input):
        """
        Args:
            video_input: (B, C, T, H, W) video frames
            skeleton_input: (B, T, num_joints, joint_dim) skeleton coordinates
            
        Returns:
            logits: (B, num_classes) action predictions
        """
        video_features    = self.video_encoder.forward_features(video_input)     # (B, embed_dim)
        skeleton_features = self.skeleton_encoder.forward_features(skeleton_input) # (B, embed_dim)
        return self.fusion_head(video_features, skeleton_features)

    def forward_video_only(self, video_input):
        video_features = self.video_encoder.forward_features(video_input)  # (B, embed_dim)
        zero_skeleton  = torch.zeros_like(video_features)
        return self.fusion_head(video_features, zero_skeleton)

    def forward_skeleton_only(self, skeleton_input):
        skeleton_features = self.skeleton_encoder.forward_features(skeleton_input)  # (B, embed_dim)
        zero_video        = torch.zeros_like(skeleton_features)
        return self.fusion_head(zero_video, skeleton_features)

    def extract_video_features(self, video_input):
        """Extract video features without classification"""
        return self.video_encoder.forward_features(video_input)

    def extract_skeleton_features(self, skeleton_input):
        """Extract skeleton features without classification"""
        return self.skeleton_encoder.forward_features(skeleton_input)


# Factory functions for creating models with different configurations
def create_multimodal_mamba_small(num_classes=36, num_frames=8, **kwargs):
    """Small multi-modal Mamba model"""
    return MultimodalActionMamba(
        num_classes=num_classes,
        num_frames=num_frames,
        skeleton_model_size='tiny',
        fusion_model_depth=4,
        **kwargs
    )


def create_multimodal_mamba_medium(num_classes=36, num_frames=8, **kwargs):
    """Medium multi-modal Mamba model"""
    return MultimodalActionMamba(
        num_classes=num_classes,
        num_frames=num_frames,
        skeleton_model_size='medium',
        fusion_model_depth=8,
        **kwargs
    )


def create_multimodal_mamba_large(num_classes=36, num_frames=8, **kwargs):
    """Large multi-modal Mamba model"""
    return MultimodalActionMamba(
        num_classes=num_classes,
        num_frames=num_frames,
        skeleton_model_size='small',
        fusion_model_depth=12,
        **kwargs
    )
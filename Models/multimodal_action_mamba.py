import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
from VideoMamba.videomamba.video_sm.models.videomamba import videomamba_middle, videomamba_small, videomamba_tiny
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

        if video_model_size == 'tiny':
            self.video_encoder = videomamba_tiny(
                pretrained=video_pretrained,
                pretrained_path=video_pretrained_path,
                num_frames=num_frames,
                num_classes=-1,  # Return features only
            )

        elif video_model_size == 'small':
            self.video_encoder = videomamba_small(
                pretrained=video_pretrained,
                pretrained_path=video_pretrained_path,
                num_frames=num_frames,
                num_classes=-1,  # Return features only
            )
        else:
            # Video encoder - VideoMamba middle model (embed_dim=576)
            self.video_encoder = videomamba_middle(
                pretrained=video_pretrained,
                pretrained_path=video_pretrained_path,
                num_frames=num_frames,
                num_classes=-1,  # Return features only
            )
            
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
        # Extract video features
        video_features = self.video_encoder.forward_features(video_input, is_embedding=True)  # (B, T*H*W, embed_dim)
        
        # Extract skeleton features
        skeleton_features = self.skeleton_encoder.forward_features(skeleton_input, is_embedding=True)  # (B, T*num_joints, embed_dim)
        
        # Fuse and classify
        logits = self.fusion_head(video_features, skeleton_features)  # (B, num_classes)
        
        return logits

    def forward_video_only(self, video_input):
        """
        Forward pass using only video input.
        
        Args:
            video_input: (B, C, T, H, W) video frames
            
        Returns:
            logits: (B, num_classes)
        """
        video_features = self.video_encoder.forward_features(video_input, is_embedding=True)
        # Use zero skeleton features
        zero_skeleton = torch.zeros_like(video_features)
        logits = self.fusion_head(video_features, zero_skeleton)
        return logits

    def forward_skeleton_only(self, skeleton_input):
        """
        Forward pass using only skeleton input.
        
        Args:
            skeleton_input: (B, T, num_joints, joint_dim) skeleton coordinates
            
        Returns:
            logits: (B, num_classes)
        """
        skeleton_features = self.skeleton_encoder.forward_features(skeleton_input, is_embedding=True)
        # Use zero video features
        zero_video = torch.zeros_like(skeleton_features)
        logits = self.fusion_head(zero_video, skeleton_features)
        return logits

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
import torch
import torch.nn as nn
from functools import partial
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath

from VideoMamba.videomamba.video_sm.models.videomamba import Block, create_block, _init_weights, segm_init_weights

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import os

MODEL_PATH = "pre_trained_models"

def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    if 'head.weight' in state_dict.keys():
        del state_dict['head.weight']
        del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

class SkeletonMamba(nn.Module):
    """
    Skeleton-based action recognition model using Mamba blocks.
    
    Processes skeleton data (joint coordinates) with variable depth Mamba blocks.
    Similar to VisionMamba but for skeleton input instead of video patches.
    """
    def __init__(
            self, 
            num_joints=42,  # Hand skeleton: 21 joints (MediaPipe) * 2
            joint_dim=3,  # 2D or 3D coordinates (x, y) or (x, y, z)
            depth=16, 
            embed_dim=192, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            num_frames=8, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            use_checkpoint=False,
            checkpoint_num=0,
            return_features_only=False,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.return_features_only = return_features_only
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.num_frames = num_frames

        # Joint embedding: project joint coordinates to embedding space
        # Input shape: (B*T, num_joints, joint_dim) -> (B*T, num_joints, embed_dim)
        self.joint_embed = nn.Linear(joint_dim, embed_dim)
        
        # Class token for skeleton
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Spatial positional embedding for joints (position in skeleton)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints + 1, embed_dim))
        
        # Temporal positional embedding across frames
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Output head setup
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # Output normalization
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # Initialize weights
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.spatial_pos_embed, std=.02)
        trunc_normal_(self.temporal_pos_embedding, std=.02)

        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"spatial_pos_embed", "temporal_pos_embedding", "cls_token"}
    
    def get_num_layers(self):
        return len(self.layers)

    def forward_features(self, x, inference_params=None, is_embedding = False):
        """
        Args:
            x: (B, T, num_joints, joint_dim) skeleton coordinates
        Returns:
            features: (B, embed_dim) class token features
        """
        B, T, J, D = x.shape
        assert J == self.num_joints, f"Expected {self.num_joints} joints, got {J}"
        assert D >= 2, f"Joint coordinates must have at least 2D (x,y), got {D}D"
        
        # Embed joint coordinates
        # (B, T, J, D) -> (B*T, J, embed_dim)
        x = x.reshape(B * T, J, D)
        x = self.joint_embed(x)  # (B*T, J, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B*T, J+1, embed_dim)
        
        # Add spatial positional embeddings (position in skeleton)
        x = x + self.spatial_pos_embed
        
        # Separate class token for later
        cls_tokens = x[:B, :1, :]
        x_without_cls = x[:, 1:]  # (B*T, J, embed_dim)
        
        # Apply temporal positional embedding
        x_without_cls = rearrange(x_without_cls, '(b t) j m -> (b j) t m', b=B, t=T)
        x_without_cls = x_without_cls + self.temporal_pos_embedding
        x_without_cls = rearrange(x_without_cls, '(b j) t m -> b (t j) m', b=B, t=T)
        
        # Recombine with class token
        x = torch.cat((cls_tokens, x_without_cls), dim=1)  # (B, 1 + T*J, embed_dim)

        x = self.pos_drop(x)

        # Mamba blocks
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if is_embedding:
            return hidden_states  # Return all token features joints for embedding extraction

        # Return only class token
        return hidden_states[:, 0, :]

    def forward(self, x, inference_params=None):
        """
        Args:
            x: (B, T, num_joints, joint_dim) skeleton coordinates
        Returns:
            logits: (B, num_classes) action predictions
        """
        features = self.forward_features(x, inference_params)
        if self.return_features_only:
            return features
        logits = self.head(self.head_drop(features))
        return logits


# Factory functions for different skeleton model variants
def skeleton_mamba_tiny(num_joints=42, num_frames=8, num_classes=1000, embed_dim=192, pretrained=False, pretrained_path=None, **kwargs):
    """Small skeleton Mamba model"""
    model = SkeletonMamba(
        num_joints=num_joints,
        joint_dim=3,
        depth=24,
        embed_dim=embed_dim,
        num_frames=num_frames,
        num_classes=num_classes,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )

    if pretrained and pretrained_path is not None:
        pretrained_path = os.path.join(MODEL_PATH, pretrained_path)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        load_state_dict(model, state_dict, center=True)

    return model


def skeleton_mamba_small(num_joints=42, num_frames=8, num_classes=1000, embed_dim=384, pretrained=False, pretrained_path=None, **kwargs):
    """Small skeleton Mamba model"""
    model = SkeletonMamba(
        num_joints=num_joints,
        joint_dim=3,
        depth=24,
        embed_dim=embed_dim,
        num_frames=num_frames,
        num_classes=num_classes,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    
    if pretrained and pretrained_path is not None:
        pretrained_path = os.path.join(MODEL_PATH, pretrained_path)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        load_state_dict(model, state_dict, center=True)

    return model


def skeleton_mamba_medium(num_joints=42, num_frames=8, num_classes=1000, embed_dim=576, pretrained=False, pretrained_path=None, **kwargs):
    """Medium skeleton Mamba model"""
    model = SkeletonMamba(
        num_joints=num_joints,
        joint_dim=3,
        depth=32,
        embed_dim=embed_dim,
        num_frames=num_frames,
        num_classes=num_classes,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )

    if pretrained and pretrained_path is not None:
        pretrained_path = os.path.join(MODEL_PATH, pretrained_path)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        load_state_dict(model, state_dict, center=True)

    return model

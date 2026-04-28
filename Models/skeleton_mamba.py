"""
Self-contained SkeletonMamba — ported from hand_action_gcn.
Uses mamba_ssm directly instead of VideoMamba's create_block.
"""
import os
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    raise ImportError(
        "mamba_ssm is required. Install with: pip install causal-conv1d mamba-ssm"
    )

MODEL_PATH = '/workspace/video-skeleton-classifier-v3/pretrained_hands'
_MODELS = {
    "tiny": os.path.join(MODEL_PATH, "tiny.pt"),
    "small": os.path.join(MODEL_PATH, "small.pt"),
}

# ---------------------------------------------------------------------------
# Mamba block (pre-norm, residual-in-residual)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, drop_path=0., rms_norm=True, residual_in_fp32=True):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = RMSNorm(embed_dim) if rms_norm else nn.LayerNorm(embed_dim)
        self.mamba = Mamba(d_model=embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, hidden_states, residual=None):
        residual = hidden_states + residual if residual is not None else hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.drop_path(self.mamba(hidden_states))
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class SkeletonMamba(nn.Module):
    """
    Skeleton-based action recognition with Mamba blocks.
    Input: (B, T, num_joints, joint_dim)
    """
    def __init__(
        self,
        num_joints=42,
        joint_dim=3,
        depth=16,
        embed_dim=192,
        num_classes=1000,
        drop_rate=0.,
        drop_path_rate=0.1,
        rms_norm=True,
        residual_in_fp32=True,
        num_frames=8,
        return_features_only=False,
        **kwargs,  # absorb unused args (ssm_cfg, bimamba, etc.) for compat
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.return_features_only = return_features_only or (num_classes <= 0)

        self.joint_embed = nn.Linear(joint_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints + 1, embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            MambaBlock(embed_dim, drop_path=dpr[i], rms_norm=rms_norm, residual_in_fp32=residual_in_fp32)
            for i in range(depth)
        ])

        self.norm_f = RMSNorm(embed_dim) if rms_norm else nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.spatial_pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embedding, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"spatial_pos_embed", "temporal_pos_embedding", "cls_token"}

    def get_num_layers(self):
        return len(self.layers)

    def forward_features(self, x, is_embedding=False, inference_params=None):
        """
        Args:
            x: (B, T, J, D)
            is_embedding: return all token features (B, 1+T*J, embed_dim) for fusion
        Returns:
            (B, embed_dim) CLS token, or (B, 1+T*J, embed_dim) when is_embedding=True
        """
        B, T, J, D = x.shape

        x = x.reshape(B * T, J, D)
        x = self.joint_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.spatial_pos_embed

        cls_tokens = x[:B, :1, :]
        x_body = x[:, 1:]

        x_body = rearrange(x_body, '(b t) j m -> (b j) t m', b=B, t=T)
        x_body = x_body + self.temporal_pos_embedding
        x_body = rearrange(x_body, '(b j) t m -> b (t j) m', b=B, t=T)

        x = torch.cat((cls_tokens, x_body), dim=1)
        x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        if is_embedding:
            return hidden_states  # (B, 1+T*J, embed_dim)
        return hidden_states[:, 0, :]  # CLS token: (B, embed_dim)

    def forward(self, x, inference_params=None):
        features = self.forward_features(x)
        if self.return_features_only:
            return features
        return self.head(features)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _load_weights(model, pretrained_path):
    state_dict = torch.load(pretrained_path, map_location='cpu')
    if 'head.weight' in state_dict:
        del state_dict['head.weight']
        del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


def skeleton_mamba_tiny(num_joints=42, num_frames=8, num_classes=1000, embed_dim=192,
                        pretrained=False, pretrained_path=None, **kwargs):
    model = SkeletonMamba(
        num_joints=num_joints, joint_dim=3, depth=10, embed_dim=embed_dim,
        num_frames=num_frames, num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, **kwargs,
    )
    if pretrained and pretrained_path is not None:
        _load_weights(model, _MODELS["tiny"])
    return model


def skeleton_mamba_small(num_joints=42, num_frames=8, num_classes=1000, embed_dim=384,
                         pretrained=False, pretrained_path=None, **kwargs):
    model = SkeletonMamba(
        num_joints=num_joints, joint_dim=3, depth=10, embed_dim=embed_dim,
        num_frames=num_frames, num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, **kwargs,
    )
    if pretrained and pretrained_path is not None:
        _load_weights(model, _MODELS["small"])
    return model


def skeleton_mamba_medium(num_joints=42, num_frames=8, num_classes=1000, embed_dim=576,
                          pretrained=False, pretrained_path=None, **kwargs):
    model = SkeletonMamba(
        num_joints=num_joints, joint_dim=3, depth=10, embed_dim=embed_dim,
        num_frames=num_frames, num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, **kwargs,
    )
    if pretrained and pretrained_path is not None:
        _load_weights(model, os.path.join(MODEL_PATH, pretrained_path))
    return model

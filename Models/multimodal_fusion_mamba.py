import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from VideoMamba.videomamba.video_sm.models.videomamba import Block, create_block, _init_weights, segm_init_weights

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class MultimodalMambaFusion(nn.Module):
    """
    Fusion module that combines skeleton and video embeddings using Mamba blocks.
    
    The fusion strategy:
    1. Takes embeddings from skeleton and video encoders (both of size embed_dim)
    2. Applies trainable modality embeddings to mark the data type (skeleton vs video)
    3. Concatenates and sums embeddings with learned fusion weights
    4. Processes through fusion Mamba blocks
    5. Outputs unified representation for classification
    """
    def __init__(
            self, 
            embed_dim=576,  # Should match video embedding dimension
            fusion_depth=8, 
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
            fc_drop_rate=0., 
            fusion_strategy='weighted',  # Options: 'new', 'average', 'weighted', 'context'
            device=None,
            dtype=None,
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Trainable modality embeddings
        # These help the model distinguish and weight different data types
        self.video_modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.skeleton_modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if fusion_strategy == 'new':
            # Learnable class token for fusion
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if fusion_strategy == 'weighted':
            # Scalar mixing weight for cls token: controls blend between video and skeleton cls
            # Using sigmoid to keep it between 0 and 1
            self.cls_mix_alpha = nn.Parameter(torch.tensor(0.5))
        
        elif fusion_strategy == 'context':
            self.alpha_mamba = create_block(
                                embed_dim,
                                ssm_cfg=ssm_cfg,
                                norm_epsilon=norm_epsilon,
                                rms_norm=rms_norm,
                                residual_in_fp32=residual_in_fp32,
                                fused_add_norm=fused_add_norm,
                                bimamba=bimamba,
                                **factory_kwargs,
                                )
            self.alpha_head = nn.Linear(embed_dim, 1)


        # Output head setup
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Fusion Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, fusion_depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.fusion_layers = nn.ModuleList(
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
                for i in range(fusion_depth)
            ]
        )
        
        # Output normalization
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # Initialize weights
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.video_modality_embed, std=.02)
        trunc_normal_(self.skeleton_modality_embed, std=.02)

        self.apply(
            partial(
                _init_weights,
                n_layer=fusion_depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.fusion_layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {"video_modality_embed", "skeleton_modality_embed"}
        if self.fusion_strategy == 'weighted':
            no_decay.add("cls_mix_alpha")
        if self.fusion_strategy == 'new':
            no_decay.add("cls_token")
        return no_decay
    
    def get_num_layers(self):
        return len(self.fusion_layers)

    def forward_fusion(
        self, 
        video_features: Tensor,  # (B, T*H*W, embed_dim)
        skeleton_features: Tensor,  # (B, T*num_joints, embed_dim)
        inference_params=None
    ):
        """
        Fuse video and skeleton embeddings.
        
        Args:
            video_features: (B, T*H*W, embed_dim) from video encoder
            skeleton_features: (B, T*num_joints, embed_dim) from skeleton encoder
            inference_params: optional inference parameters
            
        Returns:
            fused_features: (B, embed_dim) fused embedding
        """
        B = video_features.shape[0]
        
        video_cls_tokens = video_features[:, :1, :]  # (B, 1, embed_dim)
        video_features_without_cls = video_features[:, 1:]  # (B, tokens, embed_dim)

        skeleton_cls_tokens = skeleton_features[:, :1, :]  # (B, 1, embed_dim)
        skeleton_features_without_cls = skeleton_features[:, 1:]  # (B, tokens, embed_dim)


        # Add modality embeddings
        video_with_modality = video_features_without_cls + self.video_modality_embed  # (B, T*H*W, embed_dim)
        skeleton_with_modality = skeleton_features_without_cls + self.skeleton_modality_embed  # (B, T*num_joints, embed_dim)
        
        concatenated = torch.cat([video_with_modality, skeleton_with_modality], dim=1)  # (B, T*H*W + T*num_joints, embed_dim)

        cls_token = None

        if self.fusion_strategy == 'new':
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)

        elif self.fusion_strategy == 'average':
            # Simple average of cls tokens
            cls_token = (video_cls_tokens + skeleton_cls_tokens) / 2  # (B, 1, embed_dim)
        
        elif self.fusion_strategy == 'weighted':
            # Add class token - mix video and skeleton cls tokens using learned alpha
            alpha = torch.sigmoid(self.cls_mix_alpha)  # Ensures value is between 0 and 1
            cls_token = video_cls_tokens * alpha + skeleton_cls_tokens * (1 - alpha)  # (B, 1, embed_dim)
        
        elif self.fusion_strategy == 'context':
            # Use a Mamba block to compute a context-aware alpha for blending cls tokens
            alpha_tokens, residual = self.alpha_mamba(concatenated, None)  # (B, N, D)
            alpha_tokens = alpha_tokens + residual
            alpha_feat = alpha_tokens.mean(dim=1)  # (B, D)
            alpha = torch.sigmoid(self.alpha_head(alpha_feat))  # (B, 1)
            alpha = alpha.unsqueeze(-1)  # (B, 1, 1)
            cls_token = video_cls_tokens * alpha + skeleton_cls_tokens * (1 - alpha)  # (B, 1, embed_dim)
            
        concatenated = torch.cat((cls_token, concatenated), dim=1)  # (B, total_tokens, embed_dim)
        
        residual = None
        hidden_states = concatenated
        for idx, layer in enumerate(self.fusion_layers):
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

        # Return (B, embed_dim)
        return hidden_states[:, 0, :]  # Return the class token embedding

    def forward(self, video_features: Tensor, skeleton_features: Tensor, inference_params=None):
        """
        Args:
            video_features: (B, embed_dim) from video encoder
            skeleton_features: (B, embed_dim) from skeleton encoder
            
        Returns:
            logits: (B, num_classes) action predictions
        """
        fused_features = self.forward_fusion(video_features, skeleton_features, inference_params)
        logits = self.head(self.head_drop(fused_features))
        return logits
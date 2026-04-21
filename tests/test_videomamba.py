"""
Ultra-minimal Video Mamba test - one command to rule them all
"""


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from VideoMamba.videomamba.video_sm.models import (
    videomamba_tiny,
    videomamba_small,
    videomamba_middle,
    )
    
    model = videomamba_middle(pretrained=True,num_classes=10)
    model.to(device)
    
    x = torch.randn(1, 3, 32, 224, 224)  # (B, T, C, H, W)
    x = x.to(device)
    
    y = model(x)
    print(f"✓ SUCCESS! Input: {x.shape} → Output: {y.shape}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Install Video Mamba: git clone https://github.com/OpenGVLab/VideoMamba.git && cd VideoMamba && pip install -e .")
    print(f"✗ Test failed: {e}")
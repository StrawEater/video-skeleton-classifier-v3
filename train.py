#!/usr/bin/env python3
"""
Train a skeleton, video, or multimodal model from a YAML config.

Single run (single GPU):
    python train.py configs/oakink2_skeleton.yaml

Single run (multi-GPU):
    torchrun --nproc_per_node=2 train.py configs/oakink2_skeleton.yaml

Grid sweep (multi-GPU):
    torchrun --nproc_per_node=2 train.py configs/sweeps/skeleton.yaml
"""
import os
import sys

import torch.distributed as dist

from training.utils import load_config, is_sweep, expand_sweep
from training.trainer import Trainer


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])

    # Initialize the process group once for the entire run (sweep or single).
    # torchrun sets RANK; plain python does not.
    distributed = int(os.environ.get('RANK', -1)) != -1
    if distributed:
        dist.init_process_group(backend='nccl')

    is_main = (not distributed) or (dist.get_rank() == 0)

    try:
        if is_sweep(cfg):
            runs = expand_sweep(cfg)
            if is_main:
                print(f"Sweep: {len(runs)} runs")
            for i, run_cfg in enumerate(runs, 1):
                if is_main:
                    print(f"\n[{i}/{len(runs)}] {run_cfg['experiment']['name']}")
                Trainer(run_cfg, load_pretrained=True).train()
                # Barrier ensures all ranks finish before the next Trainer is built.
                if distributed:
                    dist.barrier()
        else:
            Trainer(cfg, load_pretrained=True).train()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

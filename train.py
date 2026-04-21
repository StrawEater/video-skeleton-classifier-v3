#!/usr/bin/env python3
"""
Train a skeleton, video, or multimodal model from a YAML config.

Single run:
    python train.py configs/oakink2_skeleton.yaml

Grid sweep:
    python train.py configs/sweeps/datasize_skeleton.yaml
"""
import sys

from training.utils import load_config, is_sweep, expand_sweep
from training.trainer import Trainer


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])

    if is_sweep(cfg):
        runs = expand_sweep(cfg)
        print(f"Sweep: {len(runs)} runs")
        for i, run_cfg in enumerate(runs, 1):
            print(f"\n[{i}/{len(runs)}] {run_cfg['experiment']['name']}")
            trainer = Trainer(run_cfg, load_pretrained=True)
            trainer.train()
    else:
        Trainer(cfg, load_pretrained=True).train()


if __name__ == '__main__':
    main()

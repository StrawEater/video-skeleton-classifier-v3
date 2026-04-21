#!/usr/bin/env python3
"""
Evaluate trained models on the test split.

Single config:
    python evaluate.py configs/oakink2_skeleton.yaml
    python evaluate.py configs/oakink2_skeleton.yaml --checkpoint path/to/best.pth

Sweep (evaluate all runs):
    python evaluate.py configs/sweeps/datasize_skeleton.yaml

Output is printed and saved to a CSV file.
"""
import argparse
import sys

import pandas as pd

from training.utils import load_config, is_sweep, expand_sweep
from training.trainer import Trainer


def evaluate_one(run_cfg, checkpoint_path=None):
    trainer = Trainer(run_cfg, load_pretrained=False)
    try:
        results = trainer.evaluate(checkpoint_path=checkpoint_path, split='test')
    except FileNotFoundError as e:
        print(f"  Skipping {run_cfg['experiment']['name']}: {e}")
        results = None
    finally:
        trainer._cleanup()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='YAML config (single run or sweep)')
    parser.add_argument('--checkpoint', default=None, help='Specific .pth checkpoint to evaluate')
    parser.add_argument('--output', default='evaluation_results.csv', help='CSV output path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_results = []

    if is_sweep(cfg):
        runs = expand_sweep(cfg)
        print(f"Evaluating sweep: {len(runs)} runs")
        for i, run_cfg in enumerate(runs, 1):
            print(f"\n[{i}/{len(runs)}] {run_cfg['experiment']['name']}")
            r = evaluate_one(run_cfg, checkpoint_path=args.checkpoint)
            if r is not None:
                all_results.append(r)
    else:
        r = evaluate_one(cfg, checkpoint_path=args.checkpoint)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results collected.")
        sys.exit(1)

    df = pd.DataFrame(all_results).sort_values('top1', ascending=False)

    cols = ['name', 'modality', 'size', 'dataset', 'clip_len', 'top1', 'top5', 'loss', 'n_samples']
    cols = [c for c in cols if c in df.columns]
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(df[cols].to_string(index=False))

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

import copy
import itertools
from pathlib import Path

import torch
import yaml


@torch.no_grad()
def topk_accuracy(logits, targets, topk=(1, 5)):
    maxk = max(topk)
    B = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    accs = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accs.append((correct_k / B) * 100.0)
    return accs


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    """Load YAML config, resolving optional 'base' inheritance chain."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if 'base' in cfg:
        base_ref = cfg.pop('base')
        # base can be relative to the config file's directory or to cwd
        base_path = path.parent / base_ref
        if not base_path.exists():
            base_path = Path(base_ref)
        base_cfg = load_config(str(base_path))
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def is_sweep(cfg: dict) -> bool:
    return 'sweep' in cfg


def expand_sweep(cfg: dict) -> list:
    """
    Expand a sweep config into a list of individual run configs.

    Grid values can be:
      - a list:  [tiny, medium]  → label = str(value)
      - a dict:  {label: value}  → label taken from key

    Example:
      grid:
        model.size: [tiny, medium]
        dataset.max_train_samples:
          100samples: 100
          full: null
    """
    sweep = cfg['sweep']
    base_path = sweep['base']
    base_cfg = load_config(base_path)
    grid = sweep.get('grid', {})

    if not grid:
        return [base_cfg]

    keys = list(grid.keys())
    labeled_values = []
    for k in keys:
        g = grid[k]
        if isinstance(g, list):
            labeled_values.append([(str(v) if v is not None else 'full', v) for v in g])
        elif isinstance(g, dict):
            labeled_values.append([(str(label), v) for label, v in g.items()])
        else:
            labeled_values.append([(str(g), g)])

    runs = []
    for combo in itertools.product(*labeled_values):
        run_cfg = copy.deepcopy(base_cfg)
        suffix_parts = []
        for k, (label, v) in zip(keys, combo):
            parts = k.split('.')
            d = run_cfg
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = v
            suffix_parts.append(label)

        suffix = '__'.join(suffix_parts)
        run_cfg['experiment']['name'] = f"{run_cfg['experiment']['name']}__{suffix}"
        runs.append(run_cfg)

    return runs

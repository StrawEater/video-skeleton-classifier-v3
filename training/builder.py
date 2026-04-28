import os
import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class _LabelRemapDataset(Dataset):
    """Wraps a dataset and remaps non-contiguous label IDs to [0, N).
    Exposes .labels (list[int]) for weighted sampling without loading data.
    """
    def __init__(self, ds, label_map: dict, raw_labels: list):
        self.ds = ds
        self.label_map = label_map
        self.labels = [label_map[l] for l in raw_labels]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        *inputs, label = sample
        return (*inputs, self.label_map[int(label)])

# ---------------------------------------------------------------------------
# Video transform (shared across modalities)
# ---------------------------------------------------------------------------
VIDEO_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Pretrained weight filename lookup tables
# ---------------------------------------------------------------------------
_VIDEOMAMBA_WEIGHTS = {
    ('tiny',   8):  'videomamba_t16_ssv2_f8_res224.pth',
    ('small',  8):  'videomamba_s16_ssv2_f8_res224.pth',
    ('medium', 8):  'videomamba_m16_ssv2_f8_res224.pth',
    ('tiny',  32):  'videomamba_t16_breakfast_f32_res224.pth',
    ('small', 32):  'videomamba_s16_breakfast_f32_res224.pth',
    ('medium', 32): 'videomamba_m16_breakfast_mask_ft_f32_res224.pth',
}

_SKELETON_WEIGHTS = {
    ('tiny',   8):  'skeleton_tiny_8f.pth',
    ('small',  8):  'skeleton_small_8f.pth',
    ('medium', 8):  'skeleton_medium_8f.pth',
    ('tiny',  32):  'skeleton_tiny_32f.pth',
    ('small', 32):  'skeleton_small_32f.pth',
    ('medium', 32): 'skeleton_medium_32f.pth',
}


def _pretrained_path(table, size, clip_len, pretrained_dir, override=None):
    if override:
        return override
    filename = table.get((size, clip_len))
    if filename is None:
        raise ValueError(f"No pretrained weights for size={size}, clip_len={clip_len}")
    return os.path.join(os.path.expanduser(pretrained_dir), filename)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg, load_pretrained=True):
    """Build model from config. Set load_pretrained=False when evaluating from checkpoint."""
    from Models.skeleton_mamba import skeleton_mamba_tiny, skeleton_mamba_small, skeleton_mamba_medium
    from Models.multimodal_action_mamba import MultimodalActionMamba
    from VideoMamba.videomamba.video_sm.models import videomamba_tiny, videomamba_small, videomamba_middle

    modality = cfg['experiment']['modality']
    m = cfg['model']
    d = cfg['dataset']
    t = cfg['training']

    size = m.get('size', 'tiny')
    clip_len = d['clip_len']
    num_classes = d['num_classes']
    pretrained_dir = t.get('pretrained_dir', 'pretrained_models')
    num_joints = d.get('num_joints', 21) * d.get('num_hands', 2)  # total joints for model

    if modality == 'skeleton':
        _sk_factory = {
            'tiny': skeleton_mamba_tiny,
            'small': skeleton_mamba_small,
            'medium': skeleton_mamba_medium,
        }
        model = _sk_factory[size](num_joints=num_joints, num_frames=clip_len, num_classes=num_classes)
        if load_pretrained and m.get('pretrained', False):
            path = _pretrained_path(_SKELETON_WEIGHTS, size, clip_len, pretrained_dir, m.get('pretrained_path'))
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print(f"Loaded skeleton weights: {path}")
        return model

    if modality == 'video':
        _vid_factory = {
            'tiny': videomamba_tiny,
            'small': videomamba_small,
            'medium': videomamba_middle,
        }
        use_pretrained = load_pretrained and m.get('pretrained', False)
        pretrained_path = None
        if use_pretrained:
            pretrained_path = _pretrained_path(_VIDEOMAMBA_WEIGHTS, size, clip_len, pretrained_dir, m.get('pretrained_path'))
        model = _vid_factory[size](
            pretrained=use_pretrained,
            num_classes=num_classes,
            num_frames=clip_len,
            pretrained_path=pretrained_path,
        )
        return model

    if modality == 'multimodal':
        use_vid_pt = load_pretrained and m.get('video_pretrained', False)
        use_sk_pt = load_pretrained and m.get('skeleton_pretrained', False)
        vid_pt_path = None
        sk_pt_path = None
        if use_vid_pt:
            vid_pt_path = _pretrained_path(_VIDEOMAMBA_WEIGHTS, size, clip_len, pretrained_dir, m.get('video_pretrained_path'))
        if use_sk_pt:
            sk_pt_path = _pretrained_path(_SKELETON_WEIGHTS, size, clip_len, pretrained_dir, m.get('skeleton_pretrained_path'))
        model = MultimodalActionMamba(
            num_classes=num_classes,
            num_frames=clip_len,
            video_model_size=size,
            skeleton_model_size=size,
            fusion_model_depth=m.get('fusion_depth', 4),
            video_pretrained=use_vid_pt,
            video_pretrained_path=vid_pt_path,
            skeleton_pretrained=use_sk_pt,
            skeleton_pretrained_path=sk_pt_path,
            fusion_strategy=m.get('fusion_strategy', 'weighted'),
        )
        return model

    raise ValueError(f"Unknown modality: {modality}")


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _subset(dataset, max_samples, seed=42):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), max_samples))
    return Subset(dataset, indices)


def build_dataset(cfg, split):
    """
    split: 'train' | 'val' | 'test'
    Returns a Dataset (Subset when max_train_samples is set for train split).
    """
    from Datasets.oakink2_dataset import OakInkSkeletonDataset, OakInkVideoDataset, MultimodalOakInkDataset
    from Datasets.skeleton_dataset import H2OSkeletonDataset, MultimodalH2ODataset
    from Datasets.h20_dataset import H2OVideoMambaDataset

    d = cfg['dataset']
    modality = cfg['experiment']['modality']
    training = (split == 'train')
    split_path = d[f'{split}_split']
    clip_len = d['clip_len']
    num_joints = d.get('num_joints', 21)
    normalize = d.get('normalize_skeleton', True)
    jitter = d.get('with_jitter', True) and training
    dataset_name = d.get('name', 'oakink2')

    num_eval_clips = 1 if training else d.get('num_eval_clips', 1)

    if dataset_name == 'oakink2':
        if modality == 'skeleton':
            ds = OakInkSkeletonDataset(
                split_path=split_path,
                keypoints_root=d['keypoints_root'],
                clip_len=clip_len,
                num_joints=num_joints,
                normalize_skeleton=normalize,
                with_jitter=jitter,
                training=training,
                wrist_positions_root=d.get('wrist_positions_root'),
                num_eval_clips=num_eval_clips,
            )
        elif modality == 'video':
            ds = OakInkVideoDataset(
                split_path=split_path,
                frames_root=d['frames_root'],
                clip_len=clip_len,
                transform=VIDEO_TRANSFORM,
                with_jitter=jitter,
                training=training,
            )
        elif modality == 'multimodal':
            ds = MultimodalOakInkDataset(
                split_path=split_path,
                frames_root=d['frames_root'],
                keypoints_root=d['keypoints_root'],
                clip_len=clip_len,
                num_joints=num_joints,
                video_transform=VIDEO_TRANSFORM,
                normalize_skeleton=normalize,
                with_jitter=jitter,
                training=training,
                wrist_positions_root=d.get('wrist_positions_root'),
                num_eval_clips=num_eval_clips,
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")

    elif dataset_name == 'h2o':
        if modality == 'skeleton':
            ds = H2OSkeletonDataset(
                csv_path=split_path,
                skeleton_root=d['skeleton_root'],
                clip_len=clip_len,
                num_joints=num_joints,
                training=training,
                normalize_skeleton=normalize,
                with_jitter=jitter,
                skeleton_format=d.get('skeleton_format', 'txt'),
            )
        elif modality == 'video':
            ds = H2OVideoMambaDataset(
                csv_path=split_path,
                frames_root=d['frames_root'],
                clip_len=clip_len,
                transform=VIDEO_TRANSFORM,
                training=training,
                with_jitter=jitter,
            )
        elif modality == 'multimodal':
            ds = MultimodalH2ODataset(
                csv_path=split_path,
                frames_root=d['frames_root'],
                skeleton_root=d['skeleton_root'],
                clip_len=clip_len,
                num_joints=num_joints,
                video_transform=VIDEO_TRANSFORM,
                training=training,
                normalize_skeleton=normalize,
                with_jitter=jitter,
                skeleton_format=d.get('skeleton_format', 'txt'),
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    action_only = d.get('action_only', False)
    min_class_samples = d.get('min_class_samples', None)

    if action_only or min_class_samples:
        from pathlib import Path
        import pandas as pd, json

        split_root = Path(split_path).parent.parent

        # Map each original label_id to its effective class key (action str or int)
        if action_only:
            lm = json.loads((split_root / 'label_map.json').read_text())
            id_to_effective = {int(k): v['action'] for k, v in lm.items()}
        else:
            id_to_effective = {int(lid): int(lid) for lid in ds.df['label_id'].unique()}

        # Determine valid effective classes using full segment counts
        if min_class_samples:
            all_segs = pd.read_csv(split_root / 'action_segments.txt', sep='\t')
            counts = all_segs['label_id'].map(id_to_effective).value_counts()
            valid_effective = set(counts[counts >= min_class_samples].index)
        else:
            valid_effective = set(id_to_effective.values())

        # Filter rows in this split
        keep = [i for i, lid in enumerate(ds.df['label_id'])
                if id_to_effective.get(int(lid)) in valid_effective]

        # Build contiguous mapping: original label_id -> new int id
        kept_orig_labels = [int(ds.df['label_id'].iloc[i]) for i in keep]
        sorted_effective = sorted(valid_effective & {id_to_effective[l] for l in kept_orig_labels})
        effective_to_new = {e: i for i, e in enumerate(sorted_effective)}
        label_map = {lid: effective_to_new[id_to_effective[lid]]
                     for lid in set(kept_orig_labels)}

        n_classes = len(sorted_effective)
        cfg_classes = d['num_classes']
        if n_classes != cfg_classes:
            raise ValueError(
                f"num_classes mismatch: config says {cfg_classes} but "
                f"action_only={action_only}, min_class_samples={min_class_samples} "
                f"yields {n_classes} classes. Update num_classes in your config."
            )

        ds = _LabelRemapDataset(Subset(ds, keep), label_map, kept_orig_labels)
        print(f"[action_only={action_only}, min_class_samples={min_class_samples}] "
              f"{split}: {len(keep)} samples, {n_classes} classes")

    if training:
        ds = _subset(ds, d.get('max_train_samples'))

    return ds


def build_loaders(cfg, rank=0, world_size=1):
    from torch.utils.data import DistributedSampler
    t = cfg['training']
    d = cfg['dataset']
    clip_len = d['clip_len']
    modality = cfg['experiment']['modality']
    num_workers = t.get('num_workers', 8)
    distributed = world_size > 1

    batch_size = t.get('batch_size') or _default_batch_size(modality, clip_len)

    train_ds = build_dataset(cfg, 'train')
    val_ds = build_dataset(cfg, 'val')

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        train_loader  = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                                   num_workers=num_workers, pin_memory=True)
        val_loader    = DataLoader(val_ds,   batch_size=batch_size, sampler=val_sampler,
                                   num_workers=num_workers, pin_memory=True)
    elif d.get('weighted_sampling', False):
        from torch.utils.data import WeightedRandomSampler
        from collections import Counter
        labels = train_ds.labels
        counts = Counter(labels)
        weights = [1.0 / counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def build_test_loader(cfg, rank=0, world_size=1):
    from torch.utils.data import DistributedSampler
    t = cfg['training']
    d = cfg['dataset']
    clip_len = d['clip_len']
    modality = cfg['experiment']['modality']
    num_workers = t.get('num_workers', 8)

    batch_size = t.get('batch_size') or _default_batch_size(modality, clip_len)
    test_ds = build_dataset(cfg, 'test')

    if world_size > 1:
        sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        return DataLoader(test_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def _default_batch_size(modality, clip_len):
    if clip_len == 8:
        return 8 if modality == 'multimodal' else 16
    return 2 if modality == 'multimodal' else 4

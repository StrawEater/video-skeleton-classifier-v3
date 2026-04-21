import os
import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class _LabelRemapDataset(Dataset):
    """Wraps a dataset and remaps non-contiguous label IDs to [0, N)."""
    def __init__(self, ds, label_map: dict):
        self.ds = ds
        self.label_map = label_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # sample is (video, label) or (skeleton, label) or (video, skeleton, label)
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

    min_class_samples = d.get('min_class_samples', None)
    if min_class_samples:
        from pathlib import Path
        import pandas as pd
        segments_path = Path(split_path).parent.parent / 'action_segments.txt'
        if segments_path.exists():
            counts = pd.read_csv(segments_path, sep='\t')['label_id'].value_counts()
        else:
            counts = ds.df['label_id'].value_counts()
        valid_labels = sorted(counts[counts >= min_class_samples].index)
        valid_set = set(valid_labels)
        label_map = {orig: new for new, orig in enumerate(valid_labels)}
        keep = [i for i, lid in enumerate(ds.df['label_id']) if lid in valid_set]
        ds = _LabelRemapDataset(Subset(ds, keep), label_map)
        print(f"[min_class_samples={min_class_samples}] {split}: {len(keep)} samples, {len(valid_labels)} classes")

    if training:
        ds = _subset(ds, d.get('max_train_samples'))

    return ds


def build_loaders(cfg):
    t = cfg['training']
    d = cfg['dataset']
    clip_len = d['clip_len']
    modality = cfg['experiment']['modality']
    num_workers = t.get('num_workers', 8)

    batch_size = t.get('batch_size') or _default_batch_size(modality, clip_len)

    train_ds = build_dataset(cfg, 'train')
    val_ds = build_dataset(cfg, 'val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_test_loader(cfg):
    t = cfg['training']
    d = cfg['dataset']
    clip_len = d['clip_len']
    modality = cfg['experiment']['modality']
    num_workers = t.get('num_workers', 8)

    batch_size = t.get('batch_size') or _default_batch_size(modality, clip_len)
    test_ds = build_dataset(cfg, 'test')
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def _default_batch_size(modality, clip_len):
    if clip_len == 8:
        return 8 if modality == 'multimodal' else 16
    return 2 if modality == 'multimodal' else 4

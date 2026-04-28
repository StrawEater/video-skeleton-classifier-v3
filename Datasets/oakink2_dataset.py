import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


def _eval_clip_starts(start, end, clip_len, num_clips):
    """Return num_clips evenly-spaced clip start positions across [start, end]."""
    max_start = max(start, end - clip_len)
    if num_clips == 1:
        return [(start + max_start) // 2]
    return np.linspace(start, max_start, num_clips).round().astype(int).tolist()


class OakInkSkeletonDataset(Dataset):
    """
    Skeleton dataset for OakInkV2.
    Loads hand keypoints from per-scene .npy files.
    shape: (N, 2, 21, 3) — N frames, 2 hands, 21 joints, xyz.

    When wrist_positions_root is provided the wrist joint (index 0 per hand)
    is replaced with the absolute camera-space wrist position, giving the model
    both global position and wrist-relative hand shape.
    """

    def __init__(
        self,
        split_path,
        keypoints_root,
        clip_len=8,
        num_joints=21,
        num_hands=2,
        normalize_skeleton=False,
        with_jitter=True,
        training=True,
        pad_mode='repeat',
        wrist_positions_root=None,
        num_eval_clips=1,
    ):
        self.df = pd.read_csv(split_path, sep='\t')
        print(f"Loaded {len(self.df)} samples from {split_path}")
        self.keypoints_root = keypoints_root
        self.wrist_positions_root = wrist_positions_root
        self.clip_len = clip_len
        self.num_joints = num_joints
        self.num_hands = num_hands
        self.normalize_skeleton = normalize_skeleton
        self.with_jitter = with_jitter
        self.training = training
        self.pad_mode = pad_mode
        self.num_eval_clips = num_eval_clips
        self._kp_cache = {}
        self._wp_cache = {}

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self.df['label_id'].tolist()

    def _load_keypoints(self, scene_id):
        if scene_id not in self._kp_cache:
            path = os.path.join(self.keypoints_root, f"{scene_id}.npy")
            self._kp_cache[scene_id] = np.load(path)  # (N, 2, 21, 3)
        return self._kp_cache[scene_id]

    def _load_wrist_positions(self, scene_id):
        if self.wrist_positions_root is None:
            return None
        if scene_id not in self._wp_cache:
            path = os.path.join(self.wrist_positions_root, f"{scene_id}.npy")
            self._wp_cache[scene_id] = np.load(path)  # (N, 2, 3)
        return self._wp_cache[scene_id]

    def _normalize_skeleton(self, skeleton):
        if skeleton.size == 0:
            return skeleton
        valid_mask = ~(np.isnan(skeleton).any(axis=1) | (skeleton == 0).all(axis=1))
        if valid_mask.sum() == 0:
            return skeleton
        valid = skeleton[valid_mask]
        for i in range(skeleton.shape[1]):
            col = valid[:, i]
            lo, hi = col.min(), col.max()
            if hi > lo:
                skeleton[:, i] = 2 * (skeleton[:, i] - lo) / (hi - lo) - 1
            else:
                skeleton[:, i] = 0.0
        return skeleton

    def _pad_frames(self, frames, target_len):
        if len(frames) >= target_len:
            return frames[:target_len]
        if self.pad_mode == 'zero':
            pad = np.zeros_like(frames[0])
        else:
            pad = frames[-1]
        return frames + [pad] * (target_len - len(frames))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        scene_id = row['scene_id']
        label    = int(row['label_id'])
        start    = int(row['start_frame'])
        end      = int(row['end_frame'])

        if self.training:
            max_start  = max(start, end - self.clip_len + 1)
            clip_start = random.randint(start, max_start) if max_start > start else start
            return self._load_clip(scene_id, start, end, clip_start), label

        # Eval: return N evenly-spaced clips for voting
        clip_starts = _eval_clip_starts(start, end, self.clip_len, self.num_eval_clips)
        clips = torch.stack([self._load_clip(scene_id, start, end, cs) for cs in clip_starts])
        # clips: (num_eval_clips, clip_len, 42, 3) — or (1, ...) when num_eval_clips=1
        return clips, label

    def _load_clip(self, scene_id, start, end, clip_start):
        """Load a clip given a pre-determined clip_start (used by MultimodalOakInkDataset)."""
        keypoints       = self._load_keypoints(scene_id)
        wrist_positions = self._load_wrist_positions(scene_id)
        n = len(keypoints)
        frames = []
        for i in range(self.clip_len):
            jitter = random.randint(-2, 2) if (self.with_jitter and self.training) else 0
            k = min(max(clip_start + i + jitter, start), end, n - 1)
            kp = keypoints[k].copy()
            if wrist_positions is not None:
                kp[0, 0, :] = wrist_positions[k, 0, :]
                kp[1, 0, :] = wrist_positions[k, 1, :]
            kp = kp.reshape(self.num_hands * self.num_joints, 3)
            if self.normalize_skeleton:
                kp = self._normalize_skeleton(kp)
            frames.append(kp)
        frames = self._pad_frames(frames, self.clip_len)
        return torch.from_numpy(np.stack(frames)).float()


class OakInkVideoDataset(Dataset):
    """
    Video dataset for OakInkV2.
    Frames live at scenes/{scene_id}/{frame:06d}.png.
    The dataset was recorded with 4 cameras; we use one camera's frames,
    which are at stride-4 positions named: 000001, 000005, 000009, ...
    i.e. PNG for video timestep t (0-indexed) = f"{4*t + 1:06d}.png".
    start_frame / end_frame from label_split are in keypoint-frame space;
    we convert to video-timestep space by dividing by 4.
    Mirrors H2OVideoMambaDataset interface.
    """

    def __init__(
        self,
        split_path,
        frames_root,
        clip_len=8,
        transform=None,
        with_jitter=True,
        training=True,
        num_eval_clips=1,
    ):
        self.df = pd.read_csv(split_path, sep='\t')
        print(f"Loaded {len(self.df)} samples from {split_path}")
        self.frames_root = frames_root
        self.clip_len = clip_len
        self.transform = transform
        self.with_jitter = with_jitter
        self.training = training
        self.num_eval_clips = num_eval_clips

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self.df['label_id'].tolist()

    def _load_frame(self, scene_id, t):
        # t is the video timestep (0-indexed); frame name = 4*t+1 (1-indexed original frame)
        # Fall back to the last available frame if this one is missing
        while t >= 0:
            path = os.path.join(self.frames_root, scene_id, f"{4 * t + 1:06d}.jpg")
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
            t -= 1
        raise FileNotFoundError(f"No frames found for scene {scene_id}")

    def _load_clip(self, scene_id, start, end, clip_start):
        """Load a clip given a pre-determined clip_start in skeleton frame space."""
        # Convert skeleton frame space → video timestep space (stride-4)
        vid_start      = start      // 4
        vid_end        = end        // 4
        vid_clip_start = clip_start // 4
        frames = []
        for i in range(self.clip_len):
            jitter = random.randint(-2, 2) if (self.with_jitter and self.training) else 0
            t = min(max(vid_clip_start + i + jitter, vid_start), vid_end)
            frames.append(self._load_frame(scene_id, t))
        import torchvision.transforms.functional as TF
        frames = [self.transform(f) if self.transform else TF.to_tensor(f) for f in frames]
        return torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        scene_id = row['scene_id']
        label    = int(row['label_id'])
        start    = int(row['start_frame'])
        end      = int(row['end_frame'])

        if self.training:
            max_start  = max(start, end - self.clip_len * 4 + 1)  # skeleton space
            clip_start = random.randint(start, max_start) if max_start > start else start
            return self._load_clip(scene_id, start, end, clip_start), label

        clip_starts = _eval_clip_starts(start, end, self.clip_len * 4, self.num_eval_clips)
        clips = torch.stack([self._load_clip(scene_id, start, end, cs) for cs in clip_starts])
        # clips: (num_eval_clips, C, clip_len, H, W)
        return clips, label


class MultimodalOakInkDataset(Dataset):
    """
    Combined skeleton + video dataset for OakInkV2.
    Returns (video, skeleton, label) matching MultimodalH2ODataset.
    Both modalities are sampled independently from the same action window
    (same behaviour as MultimodalH2ODataset).
    """

    def __init__(
        self,
        split_path,
        frames_root,
        keypoints_root,
        clip_len=8,
        num_joints=21,
        video_transform=None,
        normalize_skeleton=False,
        with_jitter=True,
        training=True,
        wrist_positions_root=None,
        num_eval_clips=1,
    ):
        self.df = pd.read_csv(split_path, sep='\t')
        print(f"Loaded {len(self.df)} samples from {split_path}")
        self.num_eval_clips = num_eval_clips
        self.training = training

        self.video_ds = OakInkVideoDataset(
            split_path=split_path,
            frames_root=frames_root,
            clip_len=clip_len,
            transform=video_transform,
            with_jitter=with_jitter,
            training=training,
            num_eval_clips=num_eval_clips,
        )
        self.skeleton_ds = OakInkSkeletonDataset(
            split_path=split_path,
            keypoints_root=keypoints_root,
            clip_len=clip_len,
            num_joints=num_joints,
            normalize_skeleton=normalize_skeleton,
            with_jitter=with_jitter,
            training=training,
            wrist_positions_root=wrist_positions_root,
            num_eval_clips=num_eval_clips,
        )

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self.df['label_id'].tolist()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label    = int(row['label_id'])
        scene_id = row['scene_id']
        start    = int(row['start_frame'])
        end      = int(row['end_frame'])
        clip_len = self.skeleton_ds.clip_len

        if self.training:
            max_start  = max(start, end - clip_len + 1)
            clip_start = random.randint(start, max_start) if max_start > start else start
            video    = self.video_ds._load_clip(scene_id, start, end, clip_start)
            skeleton = self.skeleton_ds._load_clip(scene_id, start, end, clip_start)
            return video, skeleton, label

        # Eval: N shared clip windows for both modalities
        clip_starts = _eval_clip_starts(start, end, clip_len, self.num_eval_clips)
        videos    = torch.stack([self.video_ds._load_clip(scene_id, start, end, cs)    for cs in clip_starts])
        skeletons = torch.stack([self.skeleton_ds._load_clip(scene_id, start, end, cs) for cs in clip_starts])
        # videos:    (num_eval_clips, C, T, H, W)
        # skeletons: (num_eval_clips, T, J, D)
        return videos, skeletons, label

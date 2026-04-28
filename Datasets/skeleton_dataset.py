import os
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle

class H2OSkeletonDataset(Dataset):
    """
    Dataset for loading hand skeleton data from H2O dataset.
    """

    def __init__(
        self,
        csv_path,
        skeleton_root,
        clip_len=8,
        num_joints=21,  # MediaPipe hand skeleton
        num_hands=2,
        transform=None,
        with_jitter=True,
        training=True,
        skeleton_format='txt',
        normalize_skeleton=True,
        pad_mode='repeat',  # 'repeat', 'zero', 'edge'
    ):
        """
        Args:
            csv_path: Path to CSV with video information, split
            skeleton_root: Root directory containing skeleton data
            clip_len: Number of frames to extract
            num_joints: Number of joints in skeleton (21 for hand, 17 for COCO, etc.)
            transform: Optional transforms to apply
            training: Whether in training mode (random sampling vs center)
            skeleton_format: Format of skeleton data ('json', 'npz', or 'pkl')
            normalize_skeleton: Whether to normalize skeleton coordinates
            pad_mode: How to pad if action is shorter than clip_len
        """
        self.df = pd.read_csv(csv_path, delim_whitespace=True)
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        self.skeleton_root = skeleton_root
        self.clip_len = clip_len
        self.num_joints = num_joints
        self.num_hands = num_hands
        self.transform = transform
        self.training = training
        self.skeleton_format = skeleton_format
        self.normalize_skeleton = normalize_skeleton
        self.pad_mode = pad_mode
        self.with_jitter = with_jitter

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return (self.df['action_label'] - 1).tolist()  # 0-indexed, matches __getitem__

    def _load_txt_skeleton(self, skeleton_dir, frame_idx):
        
        """
        Load skeleton from single-line text file format.
        Expected format per frame (single line of floats):
        [L_annot_flag] + 21*3 left hand coords + [R_annot_flag] + 21*3 right hand coords
        Total values = 1 + 63 + 1 + 63 = 128
        If a hand is not annotated (flag 0) its coords will be returned as zeros.
        Returns combined skeleton array shaped (num_hands * num_joints, 3).
        """
        path = os.path.join(skeleton_dir, f"{frame_idx:06d}.txt")
        try:
            with open(path, 'r') as f:
                line = f.readline().strip()
            if not line:
                return None
            parts = [float(x) for x in line.split()]
        except (FileNotFoundError, ValueError):
            return None

        # Expected per-hand joint dim
        joint_dim = 3
        per_hand_vals = 1 + self.num_joints * joint_dim  # flag + coords
        expected = per_hand_vals * self.num_hands
        if len(parts) < expected:
            # not enough values
            return None

        hands = []
        for h in range(self.num_hands):
            start = h * per_hand_vals
            flag = int(parts[start])
            coords = parts[start + 1: start + per_hand_vals]
            if flag == 0:
                hand_arr = np.zeros((self.num_joints, joint_dim), dtype=np.float32)
            else:
                arr = np.array(coords, dtype=np.float32)
                if arr.size != self.num_joints * joint_dim:
                    # malformed -> zero
                    hand_arr = np.zeros((self.num_joints, joint_dim), dtype=np.float32)
                else:
                    hand_arr = arr.reshape(self.num_joints, joint_dim)
            hands.append(hand_arr)

        # Combine hands: left then right (or as many hands configured)
        combined = np.concatenate(hands, axis=0)  # (num_hands*num_joints, joint_dim)
        return combined

    def _load_skeleton(self, skeleton_dir, frame_idx):
        """Load skeleton data based on format"""
        if self.skeleton_format == 'txt':
            skeleton = self._load_txt_skeleton(skeleton_dir, frame_idx)
        else:
            raise ValueError(f"Unknown skeleton format: {self.skeleton_format}")
        
        return skeleton

    def _normalize_skeleton(self, skeleton):
        """Normalize skeleton coordinates to [-1, 1] range"""
        if skeleton is None or skeleton.size == 0:
            return skeleton
        
        # Get valid (non-zero/non-NaN) coordinates
        valid_mask = ~(np.isnan(skeleton).any(axis=1) | (skeleton == 0).all(axis=1))
        
        if valid_mask.sum() == 0:
            return skeleton
        
        valid_skeleton = skeleton[valid_mask]
        
        # Normalize each axis independently
        for i in range(skeleton.shape[1]):
            col = valid_skeleton[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                skeleton[:, i] = 2 * (skeleton[:, i] - col_min) / (col_max - col_min) - 1
            else:
                skeleton[:, i] = 0
        
        return skeleton

    def _pad_skeleton(self, skeletons, target_len):
        """Pad skeleton sequence to target length"""
        current_len = len(skeletons)
        
        if current_len == target_len:
            return skeletons
        elif current_len > target_len:
            return skeletons[:target_len]
        
        # Pad to target length
        if self.pad_mode == 'repeat':
            # Repeat last frame
            pad_skeleton = skeletons[-1:] if len(skeletons) > 0 else np.zeros((1, self.num_joints, skeletons[0].shape[-1]))
            padded = skeletons + [pad_skeleton] * (target_len - current_len)
        elif self.pad_mode == 'zero':
            # Zero padding
            joint_dim = skeletons[0].shape[-1] if len(skeletons) > 0 else 3
            padded = skeletons + [np.zeros((self.num_joints, joint_dim))] * (target_len - current_len)
        elif self.pad_mode == 'edge':
            # Edge padding (repeat edge values)
            pad_skeleton = skeletons[-1:] if len(skeletons) > 0 else np.zeros((1, self.num_joints, skeletons[0].shape[-1]))
            padded = skeletons + [pad_skeleton] * (target_len - current_len)
        else:
            padded = skeletons
        
        return padded

    def __getitem__(self, index):
        """
        Returns:
            skeleton: (clip_len, num_joints, joint_dim) tensor
            label: action label (0-indexed)
        """
        row = self.df.iloc[index]

        video_rel_path = row["path"]
        label = int(row["action_label"])
        start_act = int(row["start_act"])
        end_act = int(row["end_act"])

        # Construct skeleton directory path following H2O structure
        video_rel_path_token = video_rel_path.split("/")
        video_rel_path_token[0] += "_ego"
        video_rel_path = "/".join(video_rel_path_token)
        video_rel_path = os.path.join(video_rel_path, "cam4") 
        video_rel_path = os.path.join(video_rel_path, "hand_pose") 
        
        # Path to skeleton data (you might need to adjust this based on your actual structure)
        # For example: subject1_ego/h1/skeleton/ or subject1_ego/h1/hand_skeleton/
        skeleton_dir = os.path.join(self.skeleton_root, video_rel_path)

        # Determine sampling window
        max_start = end_act - self.clip_len + 1
        if max_start < start_act:
            # Action shorter than clip → pad by repetition
            start = start_act
        else:
            if self.training:
                start = random.randint(start_act, max_start)
            else:
                start = (start_act + max_start) // 2

        # Load skeleton frames
        skeletons = []
        
        for i in range(self.clip_len):
            
            jitter =  random.randint(-2, 2) if (self.with_jitter and self.training) else 0
            frame_idx = min(max(start + i + jitter, start_act), end_act)
            
            skeleton = self._load_skeleton(skeleton_dir, frame_idx)
            
            if skeleton is None:
                # Use zero skeleton if not found
                total_joints = self.num_joints * getattr(self, 'num_hands', 1)
                skeleton = np.zeros((total_joints, 3), dtype=np.float32)
            
            if self.normalize_skeleton:
                skeleton = self._normalize_skeleton(skeleton)
            
            skeletons.append(skeleton)

        # Pad if necessary
        skeletons = self._pad_skeleton(skeletons, self.clip_len)
        
        # Stack into tensor (clip_len, num_joints, joint_dim)
        skeleton_tensor = torch.from_numpy(np.stack(skeletons, axis=0)).float()
        
        if self.transform:
            skeleton_tensor = self.transform(skeleton_tensor)

        return skeleton_tensor, label - 1  # 0-indexed labels


class MultimodalH2ODataset(Dataset):
    """
    Combined dataset for video and skeleton data.
    Loads both modalities in sync.
    """
    def __init__(
        self,
        csv_path,
        frames_root,
        skeleton_root,
        clip_len=8,
        num_joints=21,
        video_transform=None,
        training=True,
        skeleton_format='txt',
        normalize_skeleton=True,
        with_jitter=True,
    ):
        """
        Args:
            csv_path: Path to CSV with video information
            frames_root: Root directory containing video frames
            skeleton_root: Root directory containing skeleton data
            clip_len: Number of frames to extract
            num_joints: Number of joints in skeleton
            video_transform: Transforms for video frames
            training: Whether in training mode
            skeleton_format: Format of skeleton data
            normalize_skeleton: Whether to normalize skeleton
        """
        self.df = pd.read_csv(csv_path, delim_whitespace=True)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        self.frames_root = frames_root
        self.skeleton_root = skeleton_root
        self.clip_len = clip_len
        self.num_joints = num_joints
        self.video_transform = video_transform
        self.training = training
        self.skeleton_format = skeleton_format
        self.normalize_skeleton = normalize_skeleton
        self.with_jitter = with_jitter
        
        # Create individual datasets
        self.video_dataset = H2OVideoMambaDataset(
            csv_path=csv_path,
            frames_root=frames_root,
            clip_len=clip_len,
            transform=video_transform,
            training=training,
            with_jitter=with_jitter,
        )
        
        self.skeleton_dataset = H2OSkeletonDataset(
            csv_path=csv_path,
            skeleton_root=skeleton_root,
            clip_len=clip_len,
            num_joints=num_joints,
            transform=None,
            training=training,
            skeleton_format=skeleton_format,
            normalize_skeleton=normalize_skeleton,
            with_jitter=with_jitter,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns:
            video_frames: (C, T, H, W) video tensor
            skeleton: (T, num_joints, joint_dim) skeleton tensor
            label: action label (0-indexed)
        """
        video_frames, label = self.video_dataset[index]
        skeleton, _ = self.skeleton_dataset[index]
        
        return video_frames, skeleton, label


# Import H2OVideoMambaDataset for multi-modal dataset
from Datasets.h20_dataset import H2OVideoMambaDataset

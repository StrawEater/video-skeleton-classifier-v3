import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class H2OVideoMambaDataset(Dataset):
    def __init__(
        self,
        csv_path,
        frames_root,
        clip_len=32,
        transform=None,
        training=True,
        with_jitter=True,
    ):
        self.df = pd.read_csv(csv_path, delim_whitespace=True)
        self.frames_root = frames_root
        self.clip_len = clip_len
        self.transform = transform
        self.training = training
        self.with_jitter = with_jitter

    def __len__(self):
        return len(self.df)

    def _load_frame(self, frame_dir, idx):
        path = os.path.join(frame_dir, f"{idx:06d}.jpg")
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        row = self.df.iloc[index]

        video_rel_path = row["path"]
        label = int(row["action_label"])
        start_act = int(row["start_act"])
        end_act = int(row["end_act"])

        video_rel_path_token = video_rel_path.split("/")
        video_rel_path_token[0] += "_ego"
        video_rel_path = "/".join(video_rel_path_token) 
        video_rel_path = os.path.join(video_rel_path, "cam4") 
        video_rel_path = os.path.join(video_rel_path, "rgb256") 

        frame_dir = os.path.join(self.frames_root, video_rel_path)

        # make sure clip fits in action window
        max_start = end_act - self.clip_len + 1
        if max_start < start_act:
            # action shorter than clip → pad by repetition
            start = start_act
        else:
            if self.training:
                start = random.randint(start_act, max_start)
            else:
                start = (start_act + max_start) // 2

        frames = []
        for i in range(self.clip_len):
            
            jitter =  random.randint(-2, 2) if (self.with_jitter and self.training) else 0
            frame_idx = min(max(start + i + jitter, start_act), end_act)
            
            img = self._load_frame(frame_dir, frame_idx)
            frames.append(img)

        if self.transform:
            frames = [self.transform(img) for img in frames]

        frames = torch.stack(frames, dim=0)
        frames = frames.permute(1, 0, 2, 3)

        return frames, label - 1 

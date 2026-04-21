import sys
sys.path.insert(0, '/home/juanb/mnt/nikola_home/video-skeleton-classifier')

DATA_ROOT = '/home/juanb/mnt/nikola_data/Proyectos/skeleton-video-classifier/DATA/OakInkV2'
SPLIT_TRAIN = f'{DATA_ROOT}/label_split/action_train.txt'
SPLIT_VAL   = f'{DATA_ROOT}/label_split/action_val.txt'
KP_ROOT     = f'{DATA_ROOT}/hand_keypoints'
FRAMES_ROOT = f'{DATA_ROOT}/scenes'

from torchvision import transforms
from Datasets.oakink2_dataset import OakInkSkeletonDataset, OakInkVideoDataset, MultimodalOakInkDataset

basic_transform = transforms.ToTensor()

# ── Skeleton dataset ─────────────────────────────────────────────────────────
print("\n=== OakInkSkeletonDataset ===")
skel_ds = OakInkSkeletonDataset(
    split_path=SPLIT_TRAIN,
    keypoints_root=KP_ROOT,
    clip_len=8,
    training=True,
)
print(f"len = {len(skel_ds)}")
skel, label = skel_ds[0]
print(f"skeleton shape: {skel.shape}  dtype: {skel.dtype}")
print(f"label: {label}")
assert skel.shape == (8, 42, 3), f"unexpected shape {skel.shape}"
assert skel.dtype.is_floating_point
print("skeleton[0]: OK")

# Test a few more samples
for i in [10, 100, len(skel_ds)-1]:
    s, l = skel_ds[i]
    assert s.shape == (8, 42, 3)
print(f"Checked 4 samples: all shapes correct")

# ── Video dataset ─────────────────────────────────────────────────────────────
print("\n=== OakInkVideoDataset ===")
vid_ds = OakInkVideoDataset(
    split_path=SPLIT_TRAIN,
    frames_root=FRAMES_ROOT,
    clip_len=8,
    transform=basic_transform,
    training=True,
)
print(f"len = {len(vid_ds)}")
video, label = vid_ds[0]
print(f"video shape: {video.shape}  dtype: {video.dtype}")
print(f"label: {label}")
assert video.shape[0] == 3 and video.shape[1] == 8, f"unexpected shape {video.shape}"
print("video[0]: OK")

# Test eval mode (deterministic)
vid_ds_eval = OakInkVideoDataset(SPLIT_VAL, FRAMES_ROOT, clip_len=8, transform=basic_transform, training=False)
v1, l1 = vid_ds_eval[0]
v2, l2 = vid_ds_eval[0]
assert (v1 == v2).all(), "eval mode should be deterministic"
print("eval determinism: OK")

# ── Multimodal dataset ────────────────────────────────────────────────────────
print("\n=== MultimodalOakInkDataset ===")
mm_ds = MultimodalOakInkDataset(
    split_path=SPLIT_TRAIN,
    frames_root=FRAMES_ROOT,
    keypoints_root=KP_ROOT,
    clip_len=8,
    video_transform=basic_transform,
    training=True,
)
print(f"len = {len(mm_ds)}")
video, skeleton, label = mm_ds[0]
print(f"video shape: {video.shape}")
print(f"skeleton shape: {skeleton.shape}")
print(f"label: {label}")
assert video.shape[0] == 3 and video.shape[1] == 8
assert skeleton.shape == (8, 42, 3)
print("multimodal[0]: OK")

print("\nAll tests passed!")

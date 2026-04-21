import torch
from manotorch.manolayer import ManoLayer

mano_root = '/home/juanb/mnt/nikola_data/Proyectos/skeleton-video-classifier/mano_v1_2'
rh_layer = ManoLayer(mano_assets_root=mano_root, side='right', use_pca=False, flat_hand_mean=False)
print('model loaded OK')

pose = torch.zeros(1, 48)
betas = torch.zeros(1, 10)
print('running forward pass...')
out = rh_layer(pose, betas)
print('forward pass OK, output:', type(out))

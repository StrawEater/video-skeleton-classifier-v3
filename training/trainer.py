import gc
import glob
import math
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from training.builder import build_model, build_loaders, build_test_loader
from training.utils import topk_accuracy


class Trainer:
    """
    Handles training and evaluation for skeleton, video, and multimodal models.
    Modality is determined by cfg['experiment']['modality'].
    """

    def __init__(self, cfg, load_pretrained=True):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modality = cfg['experiment']['modality']

        t = cfg['training']
        checkpoint_base = os.path.expanduser(t['checkpoint_dir'])
        self.checkpoint_dir = os.path.join(checkpoint_base, cfg['experiment']['name'])

        self.model = build_model(cfg, load_pretrained=load_pretrained).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Hyperparameters
        self.total_epochs = t.get('total_epochs', 200)
        self.lr = t.get('learning_rate', 2e-4)
        self.wd = t.get('weight_decay', 0.05)
        self.warmup_fraction = t.get('warmup_fraction', 0.05)
        self.grad_clip = t.get('grad_clip', 1.0)
        self.lr_patience = t.get('lr_patience', 5)
        self.lr_factor = t.get('lr_factor', 0.1)
        self.stop_patience = t.get('stop_patience', 10)
        self.skip_if_exists = t.get('skip_if_exists', True)
        self.t_max_mult = t.get('cosine_t_max_multiplier', 1.0)


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, batch):
        if self.modality in ('skeleton', 'video'):
            x, labels = batch
            return self.model(x.to(self.device, non_blocking=True)), labels.to(self.device, non_blocking=True)
        # multimodal
        video, skeleton, labels = batch
        return (
            self.model(
                video.to(self.device, non_blocking=True),
                skeleton.to(self.device, non_blocking=True),
            ),
            labels.to(self.device, non_blocking=True),
        )

    def _run_epoch(self, loader, training: bool):
        self.model.train() if training else self.model.eval()
        total_loss = total_top1 = total_top5 = 0.0
        label = 'Train' if training else 'Val'
        pbar = tqdm(loader, desc=label, leave=False)

        with torch.set_grad_enabled(training):
            for batch in pbar:
                logits, labels = self._forward(batch)
                loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()

                top1, top5 = topk_accuracy(logits, labels)
                total_loss += loss.item()
                total_top1 += top1.item()
                total_top5 += top5.item()

                if training:
                    lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{lr:.2e}"})

        n = len(loader)
        return total_loss / n, total_top1 / n, total_top5 / n

    def _build_optimizer(self, steps_per_epoch):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        warmup_epochs = max(1, int(self.total_epochs * self.warmup_fraction))
        warmup_steps = steps_per_epoch * warmup_epochs
        cosine_steps = steps_per_epoch * (self.total_epochs - warmup_epochs)
        warmup_sched = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=int(cosine_steps * self.t_max_mult), eta_min=1e-6)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        if self.skip_if_exists and os.path.exists(self.checkpoint_dir):
            print(f"Skipping {self.cfg['experiment']['name']} — checkpoint dir already exists.")
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        train_loader, val_loader = build_loaders(self.cfg)
        self._build_optimizer(len(train_loader))

        print(f"\n{'='*60}")
        print(f"Experiment : {self.cfg['experiment']['name']}")
        print(f"Modality   : {self.modality}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Dataset    : {self.cfg['dataset']['name']}  "
              f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val)")
        print(f"{'='*60}\n")

        last_best = -math.inf
        patience_lr = patience_stop = 0
        last_ckpt = None

        for epoch in range(self.total_epochs):
            train_loss, train_top1, _ = self._run_epoch(train_loader, training=True)
            val_loss, val_top1, val_top5 = self._run_epoch(val_loader, training=False)

            ep = epoch + 1
            lr = self.optimizer.param_groups[0]['lr']
            print(f"[{ep:3d}/{self.total_epochs}] "
                  f"train {train_top1:.1f}%  "
                  f"val {val_top1:.1f}% (top5 {val_top5:.1f}%)  "
                  f"lr {lr:.2e}")

            if val_top1 > last_best:
                if last_ckpt and os.path.exists(last_ckpt):
                    os.remove(last_ckpt)
                fname = f"epoch{ep:03d}_train{train_top1:.2f}_val{val_top1:.2f}.pth"
                last_ckpt = os.path.join(self.checkpoint_dir, fname)
                torch.save(self.model.state_dict(), last_ckpt)
                last_best = val_top1
                patience_lr = patience_stop = 0
                print(f"  -> saved {fname}")
            else:
                patience_lr += 1
                patience_stop += 1

            if patience_lr >= self.lr_patience:
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= self.lr_factor
                print(f"  LR reduced to {self.optimizer.param_groups[0]['lr']:.2e}")
                patience_lr = 0

            if patience_stop >= self.stop_patience:
                print(f"  Early stop at epoch {ep}.")
                break

        print(f"\nFinished: {self.cfg['experiment']['name']}")
        self._cleanup()

    def evaluate(self, checkpoint_path=None, split='test'):
        """
        Evaluate on the given split using a saved checkpoint.
        If checkpoint_path is None, uses the latest .pth in checkpoint_dir.
        Returns a results dict.
        """
        if checkpoint_path is None:
            ckpts = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            checkpoint_path = ckpts[-1]

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        loader = build_test_loader(self.cfg) if split == 'test' else build_loaders(self.cfg)[1]

        total_loss = total_top1 = total_top5 = n_samples = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval [{split}]"):
                logits, labels = self._forward(batch)
                loss = self.criterion(logits, labels)
                top1, top5 = topk_accuracy(logits, labels)
                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_top1 += top1.item() * bs
                total_top5 += top5.item() * bs
                n_samples += bs

        results = {
            'name':       self.cfg['experiment']['name'],
            'modality':   self.modality,
            'size':       self.cfg['model'].get('size', ''),
            'dataset':    self.cfg['dataset']['name'],
            'clip_len':   self.cfg['dataset']['clip_len'],
            'checkpoint': os.path.basename(checkpoint_path),
            'split':      split,
            'n_samples':  n_samples,
            'loss':       total_loss / n_samples,
            'top1':       total_top1 / n_samples,
            'top5':       total_top5 / n_samples,
        }
        print(f"[{split}] top1={results['top1']:.2f}%  top5={results['top5']:.2f}%  "
              f"loss={results['loss']:.4f}  n={n_samples}")
        return results

    def _cleanup(self):
        for attr in ('optimizer', 'scheduler'):
            if hasattr(self, attr):
                delattr(self, attr)
        del self.model, self.criterion
        torch.cuda.empty_cache()
        gc.collect()

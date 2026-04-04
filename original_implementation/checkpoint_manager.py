"""
checkpoint_manager.py
---------------------
Handles all checkpoint saving and loading for iResNet CIFAR training.
Import and use this in main_cifar.py instead of the inline save_checkpoint function.

Usage in main_cifar.py:
    from checkpoint_manager import CheckpointManager

    # After building model/optimizer:
    ckpt_manager = CheckpointManager(result_path=args.result_path, save_freq=args.save_freq)

    # Inside training loop, after validation:
    ckpt_manager.save(
        state=...,
        is_best=is_best,
        epoch=epoch + 1
    )

    # To resume training:
    start_epoch, best_acc1 = ckpt_manager.load(args.resume, model, optimizer, scheduler, args)
"""

import os
import shutil
import torch


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints during training.

    Directory layout created inside result_path:
        result_path/
            checkpoint.pth.tar          — latest epoch (overwritten each epoch)
            model_best.pth.tar          — full checkpoint at best val acc1
            weights_best.pth.tar        — state_dict only at best val acc1 (for inference)
            checkpoints/
                epoch_0010.pth.tar      — periodic snapshots (never overwritten)
                epoch_0020.pth.tar
                ...
    """

    LATEST_NAME  = 'checkpoint.pth.tar'
    BEST_NAME    = 'model_best.pth.tar'
    WEIGHTS_NAME = 'weights_best.pth.tar'
    CKPT_SUBDIR  = 'checkpoints'

    def __init__(self, result_path, save_freq=10):
        """
        Args:
            result_path (str): Root directory for all outputs.
            save_freq   (int): Save a periodic snapshot every N epochs.
                               Set to 0 to disable periodic snapshots.
        """
        self.result_path = result_path
        self.save_freq   = save_freq
        self.ckpt_dir    = os.path.join(result_path, self.CKPT_SUBDIR)

        os.makedirs(self.result_path, exist_ok=True)
        if save_freq > 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, state, is_best, epoch):
        """Save checkpoint files for the current epoch.

        Args:
            state   (dict): Dict containing at minimum:
                                'epoch', 'arch', 'state_dict',
                                'best_acc1', 'val_acc1', 'optimizer'
            is_best (bool): True if this epoch has the best val acc1 so far.
            epoch   (int) : Current epoch number (1-indexed).
        """
        latest_path = os.path.join(self.result_path, self.LATEST_NAME)

        # 1. Always overwrite the latest checkpoint (enables --resume)
        torch.save(state, latest_path)

        # 2. Best model — two files: full checkpoint + weights-only
        if is_best:
            best_path    = os.path.join(self.result_path, self.BEST_NAME)
            weights_path = os.path.join(self.result_path, self.WEIGHTS_NAME)
            shutil.copyfile(latest_path, best_path)
            torch.save(state['state_dict'], weights_path)
            print("  [CheckpointManager] New best model saved "
                  "(epoch {:d}, val_acc1={:.2f}%)".format(epoch, state['val_acc1']))

        # 3. Periodic snapshot — never overwritten
        if self.save_freq > 0 and epoch % self.save_freq == 0:
            periodic_path = os.path.join(
                self.ckpt_dir, 'epoch_{:04d}.pth.tar'.format(epoch)
            )
            shutil.copyfile(latest_path, periodic_path)
            print("  [CheckpointManager] Periodic snapshot saved: {}".format(periodic_path))

    # ------------------------------------------------------------------
    # Loading / Resuming
    # ------------------------------------------------------------------

    def load(self, resume_path, model, optimizer, scheduler=None, args=None):
        """Load a checkpoint and restore model/optimizer/scheduler state.

        Args:
            resume_path (str)            : Path to the checkpoint file.
            model       (nn.Module)      : Model to load weights into.
            optimizer   (Optimizer)      : Optimizer to restore state into.
            scheduler   (LRScheduler)    : Optional LR scheduler to restore.
            args        (Namespace)      : args object; start_epoch will be set.

        Returns:
            start_epoch (int)  : Epoch to resume from.
            best_acc1   (float): Best val acc1 recorded so far.
        """
        if not os.path.isfile(resume_path):
            print("  [CheckpointManager] No checkpoint found at '{}'".format(resume_path))
            return 0, 0

        print("  [CheckpointManager] Loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)

        start_epoch = checkpoint['epoch']
        best_acc1   = checkpoint['best_acc1']

        # Move best_acc1 to the right device if needed
        if args is not None and args.gpu is not None and torch.is_tensor(best_acc1):
            best_acc1 = best_acc1.to(args.gpu)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Restore scheduler to the correct step
        if scheduler is not None:
            scheduler.last_epoch = start_epoch - 1
            # MultiStepLR needs to be re-created with last_epoch to sync milestones;
            # caller should re-create it after this call if required (see note below).

        if args is not None:
            args.start_epoch = start_epoch

        print("  [CheckpointManager] Resumed from epoch {:d} "
              "(best_acc1={:.2f}%)".format(start_epoch, best_acc1))

        return start_epoch, best_acc1

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_checkpoints(self):
        """Print all saved checkpoints in the result directory."""
        files = []
        for fname in [self.LATEST_NAME, self.BEST_NAME, self.WEIGHTS_NAME]:
            p = os.path.join(self.result_path, fname)
            if os.path.exists(p):
                files.append((p, os.path.getsize(p)))

        if os.path.exists(self.ckpt_dir):
            for fname in sorted(os.listdir(self.ckpt_dir)):
                p = os.path.join(self.ckpt_dir, fname)
                files.append((p, os.path.getsize(p)))

        print("\n{:-<60}".format(''))
        print(" Saved checkpoints in: {}".format(self.result_path))
        print("{:-<60}".format(''))
        for path, size in files:
            print("  {:<45s}  {:>7.1f} MB".format(
                os.path.relpath(path, self.result_path), size / 1e6))
        print("{:-<60}".format(''))

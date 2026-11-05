"""
train_finetune.py — Entry point for fine-tuning HOISTFormer on our 8-class
CVAT data. Thin wrapper around HOISTFormer/train_net_video.py: registers our
datasets first, then reuses the repo's own Trainer/setup unchanged (no edits
to train_net_video.py or the model code), adding one thing on top: optional
early stopping based on a plateau in the smoothed total_loss.

Must be run with cwd = third_party/HOISTFormer (relative pretrained-weights path
in the config depends on it), which scripts/train_lopo.sh already does.

Early stopping (opt-in via env vars, off by default):
    EARLY_STOP_PATIENCE   iterations without improvement before stopping (0 = disabled)
    EARLY_STOP_MIN_DELTA  minimum drop in smoothed total_loss to count as improvement
    EARLY_STOP_WARMUP     iterations to wait before the check starts
    EARLY_STOP_CHECK_PERIOD  how often (in iterations) to evaluate the plateau condition

Detectron2's train loop is `for self.iter in range(start_iter, max_iter)`, so
mutating trainer.iter from a hook can't cut the loop short — we stop by
raising an exception from the hook and catching it here, after which
whatever periodic checkpoint was last written (PeriodicCheckpointer already
runs every SOLVER.CHECKPOINT_PERIOD iterations) is the model to use.

Usage (same flags as train_net_video.py):
    python train_finetune.py --config-file configs/hoist/my_finetune.yaml \
        --num-gpus 1
"""

import logging
import os
import sys
import os
from pathlib import Path

# HOISTFormer/detectron2 (pinned to an old numpy) still uses the np.bool /
# np.int / np.float aliases removed in numpy>=1.24. Shim them back before any
# other import touches numpy, so dataloader worker processes (forked from
# this one) inherit the patched module too.
import numpy as np  # noqa: E402

for _name, _builtin in [("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.HOIST_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import register_finetune_dataset  # noqa: F401,E402  (side effect: registers datasets)
from detectron2.engine import HookBase, default_argument_parser, launch  # noqa: E402
import train_net_video as tnv  # noqa: E402


class EarlyStopException(Exception):
    pass


class EarlyStoppingHook(HookBase):
    def __init__(self, patience, min_delta, warmup, check_period, smoothing_window=100):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.check_period = check_period
        self.smoothing_window = smoothing_window
        self.best = float("inf")
        self.best_iter = 0

    def after_step(self):
        it = self.trainer.iter
        if it < self.warmup or (it + 1) % self.check_period != 0:
            return
        history = self.trainer.storage.history("total_loss")
        smoothed = history.median(min(self.smoothing_window, len(history.values())))
        if smoothed < self.best - self.min_delta:
            self.best = smoothed
            self.best_iter = it
        elif it - self.best_iter >= self.patience:
            logging.getLogger(__name__).info(
                f"Early stopping at iter {it}: smoothed total_loss={smoothed:.4f} "
                f"has not improved on best={self.best:.4f} (iter {self.best_iter}) "
                f"for {self.patience} iterations."
            )
            raise EarlyStopException()


def build_trainer_class(patience, min_delta, warmup, check_period):
    class EarlyStoppingTrainer(tnv.Trainer):
        def build_hooks(self):
            hooks = super().build_hooks()
            if patience > 0:
                hooks.append(EarlyStoppingHook(patience, min_delta, warmup, check_period))
            return hooks
    return EarlyStoppingTrainer


def main(args):
    cfg = tnv.setup(args)

    patience = int(os.environ.get("EARLY_STOP_PATIENCE", "0"))
    min_delta = float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.01"))
    warmup = int(os.environ.get("EARLY_STOP_WARMUP", "300"))
    check_period = int(os.environ.get("EARLY_STOP_CHECK_PERIOD", "50"))

    trainer_cls = build_trainer_class(patience, min_delta, warmup, check_period)
    trainer = trainer_cls(cfg)
    trainer.resume_or_load(resume=args.resume)
    try:
        return trainer.train()
    except EarlyStopException:
        logging.getLogger(__name__).info("Stopped early; latest periodic checkpoint is the model to use.")
        return None


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

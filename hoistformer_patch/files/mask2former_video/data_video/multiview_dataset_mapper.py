"""Dataset mapper for multi-view clips (variant B).

A multi-view clip (built by scripts/multiview/build_multiview_ytvis.py) stores its frames as
(moment, view) pairs, moment-major: position = moment * n_views + view. Sampling loose
positions out of that, as the single-view mapper does, would usually give frames of different
instants from different cameras -- no two of them show the same scene, and the cross-view
constraint the clip was built for never appears in a training step.

This mapper samples whole moments instead: a few instants, and for each of them the same
subset of cameras. Within one sample a query therefore sees the same object from several
angles at the same time, which is the only thing the single-view model cannot learn.

Views are re-drawn every time a clip is sampled, so a recording with four annotated cameras
contributes many different camera pairs over training rather than one fixed pair.
"""
import logging
import random

import numpy as np
from detectron2.config import configurable

from .dataset_mapper import YTVISDatasetMapper


class MultiViewDatasetMapper(YTVISDatasetMapper):

    @configurable
    def __init__(self, *, view_num: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.view_num = view_num
        logging.getLogger(__name__).info(
            f"[MultiViewDatasetMapper] up to {view_num} views per sample, "
            f"{self.sampling_frame_num} frames total")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret["view_num"] = cfg.INPUT.MULTIVIEW_VIEW_NUM
        return ret

    def select_indices(self, dataset_dict, video_length):
        n_views = dataset_dict.get("n_views", 1)
        if not self.is_train or n_views <= 1:
            return super().select_indices(dataset_dict, video_length)

        n_moments = video_length // n_views
        k = min(self.view_num, n_views)
        n_sampled_moments = max(1, self.sampling_frame_num // k)

        views = sorted(random.sample(range(n_views), k))

        ref = random.randrange(n_moments)
        lo = max(0, ref - self.sampling_frame_range)
        hi = min(n_moments, ref + self.sampling_frame_range + 1)
        pool = [m for m in range(lo, hi) if m != ref]
        extra = (np.random.choice(pool, n_sampled_moments - 1, replace=len(pool) < n_sampled_moments - 1).tolist()
                 if pool and n_sampled_moments > 1 else [])
        moments = sorted(set(extra + [ref])) or [ref]

        selected = [m * n_views + v for m in moments for v in views]
        if self.sampling_frame_shuffle:
            random.shuffle(selected)
        return selected

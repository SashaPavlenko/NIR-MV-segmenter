# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_video_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation", "brightness_up", "contrast_up"

    # Per-clip metadata conditioning (thesis fine-tune ablations). Off by default,
    # zero-initialized when on, so enabling it starts identical to the unconditioned
    # model and only diverges once fine-tuning updates the embeddings.
    # activity_id / camera_pos_id come from the YTVIS "videos" entries (see
    # thesis/scripts/finetune/clip_metadata.py) and are threaded through by
    # mask2former_video/data_video/datasets/hoist.py; -1 (absent/unknown) maps to
    # the last embedding row.
    cfg.MODEL.MASK_FORMER.USE_ACTIVITY_COND = False
    cfg.MODEL.MASK_FORMER.NUM_ACTIVITY_CLASSES = 4  # idle, drill, scalpel, screwdriver
    cfg.MODEL.MASK_FORMER.USE_CAMERA_COND = False
    cfg.MODEL.MASK_FORMER.NUM_CAMERA_POSITIONS = 3  # table_up (cam1/2), front_angled (cam3), top_down (cam4)

    # Per-class loss re-weighting (thesis ablation "toolw"): one float per real
    # class in categories.CATEGORIES order; [] = off. Scales the class's CE
    # weight and the mask/dice loss of its instances (see criterion.py).
    cfg.MODEL.MASK_FORMER.CLASS_LOSS_WEIGHTS = []

    # Horizontal flip swaps physical left/right, so a flipped "hand_left" is
    # visually a right hand. Pairs of contiguous class ids to swap whenever a
    # clip is h-flipped (thesis ablation "flip_swap"); [] = keep labels as-is
    # (the original, label-noisy behaviour). E.g. [[0, 1], [2, 3]] for
    # hand_left<->hand_right, haptic_left<->haptic_right.
    cfg.INPUT.HFLIP_SWAP_CLASS_PAIRS = []

    # Multi-view training (thesis variant B): how many cameras of the same instant
    # go into one clip. 0 keeps the single-view behaviour. The clip's frames are
    # (moment, view) pairs, so SAMPLING_FRAME_NUM is split as
    # moments = SAMPLING_FRAME_NUM // MULTIVIEW_VIEW_NUM.
    cfg.INPUT.MULTIVIEW_VIEW_NUM = 0

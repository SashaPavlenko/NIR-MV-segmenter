"""Soft predictions -> binary masks: window assembly, ensemble of two models, post-processing.

The soft predictions are written by infer_soft.py (one .npz per view). The binary masks are
written in the "raw" format that crossview_consensus.py and postprocess_and_export.py read:
masks (N, T, H, W) uint8, labels (N,) int32 (index into categories.CATEGORIES), scores (N,).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories import CATEGORIES  # noqa: E402

# classes that get the small-component filter (same set as categories.TOOL_CATEGORIES)
CC_CLASSES = {"haptic_left", "haptic_right", "scalpel", "screwdriver", "drill"}
FRAME_SIZE = 384


def load_soft(path):
    z = np.load(path)
    return {"starts": z["starts"], "scores": z["scores"], "probs": z["probs"], "T": int(z["T"])}


def assemble(soft, mode="overlap", thr=0.5, win=16):
    """Per-class probability maps (C, T, H, W) float32 in [0, 1] and per-frame presence (C, T).

    nonoverlap: windows at 0, 16, 32, ... (original single-view pipeline); a class is present
                in a window if its best score exceeds thr.
    overlap:    every stored window (stride 8); a frame averages the probabilities of all
                windows that detect the class.
    """
    starts, scores, probs, T = soft["starts"], soft["scores"], soft["probs"], soft["T"]
    C = scores.shape[1]
    H, W = probs.shape[-2:]
    acc = np.zeros((C, T, H, W), np.float32)
    cnt = np.zeros((C, T), np.float32)
    sel = [i for i, s in enumerate(starts) if mode == "overlap" or s % win == 0]
    for i in sel:
        s = int(starts[i])
        n = min(win, T - s)
        for c in range(C):
            if scores[i, c] > thr:
                acc[c, s:s + n] += probs[i, c, :n].astype(np.float32) / 255.0
                cnt[c, s:s + n] += 1
    present = cnt > 0
    acc[present] /= cnt[present][:, None, None]
    return acc, present


def ensemble(p1, q1, p2, q2, n1=1):
    """Running average of two models: p1 is already the mean of n1 models, p2 is one more model.

    A class counts as present in a frame if either model detects it there. Where only one
    model detects it, its probability is used as it is.
    """
    q = q1 | q2
    w1, w2 = q1.astype(np.float32) * n1, q2.astype(np.float32)
    den = np.maximum(w1 + w2, 1)
    p = (p1 * w1[..., None, None] + p2 * w2[..., None, None]) / den[..., None, None]
    return p.astype(np.float32), q


def drop_small(mask, ratio=0.15):
    """Drop connected components smaller than `ratio` of the largest one in the frame."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= ratio * areas.max()) + 1
    return np.isin(lab, keep)


def binarize(prob, present, cut=0.5, post=True):
    """dict class -> (T, H, W) bool, with the post-processing of the single-view pipeline."""
    out = {}
    for c, name in enumerate(CATEGORIES):
        m = (prob[c] > cut) & present[c][:, None, None]
        if post and name in CC_CLASSES:
            for t in np.flatnonzero(m.any(axis=(1, 2))):
                m[t] = drop_small(m[t])
        out[name] = m
    return out


def to_raw_npz(masks, path):
    """Write binary masks in the raw format (input of crossview_consensus.py)."""
    labs = [i for i, c in enumerate(CATEGORIES) if masks[c].any()]
    if labs:
        arr = np.stack([masks[CATEGORIES[i]] for i in labs]).astype(np.uint8)
    else:
        arr = np.zeros((0, 1, FRAME_SIZE, FRAME_SIZE), np.uint8)
    np.savez_compressed(path, masks=arr, labels=np.array(labs, np.int32),
                        scores=np.ones(len(labs), np.float32))


def from_raw_npz(path, T):
    """Read the raw format back into dict class -> (T, H, W) bool."""
    z = np.load(path)
    out = {c: np.zeros((T, FRAME_SIZE, FRAME_SIZE), bool) for c in CATEGORIES}
    for m, l in zip(z["masks"], z["labels"]):
        out[CATEGORIES[int(l)]] |= m.astype(bool)
    return out

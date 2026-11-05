"""Stereo geometry of one Stereo IR 170 device from its stored LEAP_DISTORTION_MATRIX.

The 64x64 grid maps a normalised ray coordinate u in [0, 1] to the normalised raw-image
position that sees this ray (LeapC convention: ray slope s = (u - 0.5) * 8). Resampling both
eyes onto a common grid of ray slopes gives a rectified pair: a scene point appears on the same
row in both eyes and is shifted horizontally by d = f * B / Z.
"""
from __future__ import annotations

import struct
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

GRID_DIR = paths.GRID_DIR   # LEAP_DISTORTION_MATRIX per device and eye, the same in every recording
N = 64
SLOPE_RANGE = 4.0
SLOPE_MAX = 2.5          # rectified image covers slopes +-2.5 (+-68 deg)
SIZE = 640               # rectified image size -> f = 127.8 px
RAW = 384
BASELINE_CM = 7.8        # estimated from the rig (see thesis), only used for metric depth
# residual vertical offset right vs left eye (px at f = 95.9), measured with SIFT matches
DY = {1: -1.15, 2: 0.18, 3: -0.31, 4: 1.46}


def read_grid(path: Path) -> np.ndarray:
    b = Path(path).read_bytes()
    magic, _, n, _, _, size, _ = struct.unpack("<IIIIQII", b[:32])
    assert magic == 0x5453444C and n == N
    return np.frombuffer(b[32:32 + size], dtype=np.float32).reshape(N, N, 2).copy()


class Eye:
    """Rectification maps of one eye: raw (384^2) <-> rectified slope image (SIZE^2)."""

    def __init__(self, dev: int, side: str):
        g = read_grid(GRID_DIR / str(dev) / f"distortion_{side}.bin")
        s = np.linspace(-SLOPE_MAX, SLOPE_MAX, SIZE, dtype=np.float32)
        gx = (s / (2 * SLOPE_RANGE) + 0.5) * (N - 1)
        gxx, gyy = np.meshgrid(gx, gx)
        if side == "right":
            gyy = gyy - DY[dev] / 95.9 / (2 * SLOPE_RANGE) * (N - 1)   # vertical correction
        rx = cv2.remap(g[..., 0], gxx.astype(np.float32), gyy.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        ry = cv2.remap(g[..., 1], gxx.astype(np.float32), gyy.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        self.map_x = (rx * (RAW - 1)).astype(np.float32)   # rect pixel -> raw x
        self.map_y = (ry * (RAW - 1)).astype(np.float32)
        self.f = (SIZE - 1) / (2 * SLOPE_MAX)
        self.c = (SIZE - 1) / 2
        ok = (rx >= 0) & (ry >= 0)
        vv, uu = np.mgrid[0:SIZE, 0:SIZE]
        tree = cKDTree(np.stack([self.map_x[ok], self.map_y[ok]], 1))
        py, px = np.mgrid[0:RAW, 0:RAW]
        dist, idx = tree.query(np.stack([px.ravel(), py.ravel()], 1).astype(np.float32))
        # raw pixel -> rect coordinates (nearest rect sample, < 0.5 rect px error)
        self.inv_u = uu[ok][idx].reshape(RAW, RAW).astype(np.float32)
        self.inv_v = vv[ok][idx].reshape(RAW, RAW).astype(np.float32)
        self.inv_ok = (dist < 1.0).reshape(RAW, RAW)
        self.rect_ok = ok

    def to_rect(self, img, interp=cv2.INTER_LINEAR):
        return cv2.remap(img, self.map_x, self.map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def to_raw(self, rect, interp=cv2.INTER_LINEAR):
        out = cv2.remap(rect, self.inv_u, self.inv_v, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[~self.inv_ok] = 0
        return out


@lru_cache(maxsize=8)
def eye(dev: int, side: str) -> Eye:
    return Eye(dev, side)


def sgm(Lr, Rr, nd=128, block=5):
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=block,
                              P1=8 * block * block, P2=32 * block * block, uniquenessRatio=5,
                              speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                              mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    Lp = cv2.copyMakeBorder(Lr, 0, 0, nd, 0, cv2.BORDER_CONSTANT, value=0)
    Rp = cv2.copyMakeBorder(Rr, 0, 0, nd, 0, cv2.BORDER_CONSTANT, value=0)
    d = m.compute(Lp, Rp).astype(np.float32)[:, nd:] / 16
    d[d <= 0] = 0
    return d


def right_from_left(dL):
    """Forward-warp a left-referenced disparity map into the right eye (z-buffer: larger d wins)."""
    H, W = dL.shape
    dR = np.zeros_like(dL)
    v, u = np.nonzero(dL > 0)
    d = dL[v, u]
    ur = np.round(u - d).astype(int)
    ok = (ur >= 0) & (ur < W)
    v, ur, d = v[ok], ur[ok], d[ok]
    order = np.argsort(d)                  # write small first, large (near) last
    dR[v[order], ur[order]] = d[order]
    return dR

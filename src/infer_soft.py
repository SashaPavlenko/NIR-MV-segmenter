"""Windowed inference that keeps soft masks.

    cd third_party/HOISTFormer   # configs/ and the model code are found from here
    python ../../src/infer_soft.py --config-file configs/hoist/v4_combo.yaml \
        --weights ../../weights/combo_nir/fold1.pth --out ../../outputs/x/soft_nir


For every window of WIN frames (start every STRIDE frames) and every class, the best
(query, class) pair of the top-10 is stored with its score and its mask probabilities
(sigmoid of the mask logits, uint8, values below 0.04 set to 0). With STRIDE = WIN / 2,
the non-overlapping windows of the original pipeline are a subset of the stored ones.

Output per job: <out>/job_<id>.npz
    starts (W,), scores (W, C) float32 (0 if class not among the top-10),
    probs  (W, C, WIN, H, W) uint8 -- only windows/classes with score > 0.05 are filled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

HOIST = paths.HOIST_DIR
sys.path.insert(0, str(HOIST))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

C = 8


def soft_inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
    scores = F.softmax(pred_cls, dim=-1)[:, :-1]
    labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
        self.num_queries, 1).flatten(0, 1)
    scores_per_image, topk = scores.flatten(0, 1).topk(10, sorted=False)
    labels_per_image = labels[topk]
    pred_masks = pred_masks[topk // self.sem_seg_head.num_classes]
    pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
    pred_masks = F.interpolate(pred_masks, size=(output_height, output_width), mode="bilinear",
                               align_corners=False)
    probs = (pred_masks.sigmoid() * 255).round().to(torch.uint8)
    return {"scores": scores_per_image.cpu().numpy(), "labels": labels_per_image.cpu().numpy(),
            "probs": probs.cpu().numpy()}


def load(cfg_file, weights):
    from demo_video.predictor import VisualizationDemo
    demo = VisualizationDemo(cfg_file, weights)
    model = demo.predictor.model
    model.inference_video = types.MethodType(soft_inference_video, model)
    return demo.predictor


def frames_of(job, data_root: Path, ext: str):
    run, cam = job["task_name"].split("__")
    dev, side = cam[3], cam.split("_")[1]
    d = data_root / run / "images" / dev / side
    out = []
    for n in job["frame_names"]:
        img = cv2.imread(str(d / (n[:-4] + ext)))
        if img is None:
            raise IOError(d / (n[:-4] + ext))
        out.append(img)
    return out


def infer(predictor, frames, win, stride):
    T = len(frames)
    H, W = frames[0].shape[:2]
    starts = list(range(0, max(T - win, 0) + 1, stride))
    if starts[-1] + win < T:
        starts.append(T - win if T >= win else 0)
    # make sure the non-overlapping grid 0, win, 2win, ... is contained (last one may be short)
    grid = list(range(0, T, win))
    starts = sorted(set(starts) | set(grid))
    scores = np.zeros((len(starts), C), np.float32)
    probs = np.zeros((len(starts), C, win, H, W), np.uint8)
    for i, s in enumerate(starts):
        sub = frames[s:s + win]
        with torch.no_grad():
            out = predictor(sub)
        for c in range(C):
            idx = np.where(out["labels"] == c)[0]
            if len(idx) == 0:
                continue
            b = idx[np.argmax(out["scores"][idx])]
            scores[i, c] = out["scores"][b]
            if scores[i, c] > 0.05:
                p = out["probs"][b]
                p[p < 10] = 0
                probs[i, c, :len(sub)] = p
    return np.array(starts), scores, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--jobs", nargs="*", type=int, default=None, help="default: every view in --job-frames")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=paths.RECORDINGS_DIR,
                    help="recordings (NIR model) or the disparity frames of precompute_depth.py (depth model)")
    ap.add_argument("--ext", default=".bmp", help=".bmp for recordings, .png for disparity frames")
    ap.add_argument("--job-frames", type=Path, default=paths.JOB_FRAMES)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = {j["job_id"]: j for j in json.load(open(a.job_frames))}
    predictor = load(a.config_file, a.weights)
    for jid in (a.jobs or sorted(jobs)):
        p = a.out / f"job_{jid}.npz"
        if p.exists():
            continue
        t0 = time.time()
        frames = frames_of(jobs[jid], a.data_root, a.ext)
        starts, scores, probs = infer(predictor, frames, a.win, a.stride)
        np.savez_compressed(p, starts=starts, scores=scores, probs=probs, T=len(frames))
        print(f"job {jid}: {len(frames)} frames, {len(starts)} windows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

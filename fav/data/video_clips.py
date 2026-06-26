"""Real-video training source: stream consecutive-frame tuples from clip folders.

A ``video_dir`` contains one subdirectory per clip, each holding consecutive
frame images. ``VideoClipsSource.sample(num, batch)`` draws ``batch`` random
(clip, start) windows of ``num+1`` consecutive frames, takes a common random
crop per window, and computes the backward flow + reliability on the fly with a
``FlowEstimator`` — yielding the same ``(imgsList, flowList, certList)`` rollout
tuple shape as the synthetic sources (frames are raw RGB; the training loop
preprocesses).
"""

from __future__ import annotations

from pathlib import Path

import torch

from fav.data.io_utils import list_frames, load_rgb
from fav.flow.estimator import FlowEstimator, estimate_at_friendly_size
from fav.occlusion.consistency import compute_reliability
from fav.warp.flow_io import uv_to_dydx


def build_video_tuple_batched(frames, estimator: FlowEstimator, use_structure: bool = True):
    """Build a batched training tuple from estimated flow/occlusion.

    Args:
        frames: list over time (len ``num+1``) of ``(b,3,H,W)`` RGB tensors.
    Returns:
        ``(imgsList, flowList, certList)`` with ``imgsList[i]=(b,3,H,W)``,
        ``flowList[i]=(b,2,H,W)`` ``(dy,dx)``, ``certList[i]=(b,1,H,W)`` in [0,1].
    """
    num = len(frames) - 1
    flow_list, cert_list = [], []
    for i in range(num):
        prev, cur = frames[i], frames[i + 1]
        # The estimator may run on a different device (e.g. RAFT on MPS); bring the
        # flow back to the frames' device so the downstream occlusion math (which
        # mixes flow with the content image) stays on a single device.
        backward = estimate_at_friendly_size(estimator, cur, prev).to(prev.device)  # cur->prev
        forward = estimate_at_friendly_size(estimator, prev, cur).to(prev.device)   # prev->cur
        certs = []
        for e in range(cur.shape[0]):
            content = cur[e] if use_structure else None
            rel = compute_reliability(backward[e], forward[e], content_image=content)
            certs.append(rel / 255.0)
        cert = torch.stack(certs).unsqueeze(1)  # (b,1,H,W)
        flow_list.append(uv_to_dydx(backward))
        cert_list.append(cert)
    return list(frames), flow_list, cert_list


class VideoClipsSource:
    def __init__(self, video_dir, estimator: FlowEstimator, crop: int = 256,
                 use_structure: bool = True, generator=None):
        self.root = Path(video_dir)
        self.estimator = estimator
        self.crop = crop
        self.use_structure = use_structure
        self.generator = generator
        # Each clip = a subdirectory with >= 2 frames (fall back to the root if it
        # directly contains frames).
        self.clips = []
        for sub in sorted(p for p in self.root.iterdir() if p.is_dir()):
            frames = list_frames(sub)
            if len(frames) >= 2:
                self.clips.append(frames)
        if not self.clips:
            root_frames = list_frames(self.root)
            if len(root_frames) >= 2:
                self.clips.append(root_frames)
        if not self.clips:
            raise FileNotFoundError(f"no usable clips (>=2 frames) under {self.root}")

    def _rand(self, high: int) -> int:
        return int(torch.randint(0, high, (1,), generator=self.generator).item())

    def _load_window(self, num: int):
        """Load ``num+1`` consecutive frames (each (3,H,W)) from a random clip+crop.

        Returns ``(frames, real_len)`` where ``real_len`` is the number of genuine
        consecutive frames (``< num+1`` only when no clip is long enough); the
        caller marks transitions past ``real_len`` as occluded so the repeated-frame
        padding never trains spurious zero-motion / perfect-consistency samples.
        """
        candidates = [c for c in self.clips if len(c) >= num + 1]
        clip = candidates[self._rand(len(candidates))] if candidates else max(self.clips, key=len)
        eff = min(num + 1, len(clip))
        start = self._rand(len(clip) - eff + 1)
        frames = [load_rgb(clip[start + k], batched=False) for k in range(eff)]
        real_len = len(frames)
        # Repeat the last frame if the clip was shorter than requested (rare).
        while len(frames) < num + 1:
            frames.append(frames[-1].clone())
        # Common random crop across the window so motion is coherent.
        _, h, w = frames[0].shape
        size = self.crop
        if h < size or w < size:
            frames = [
                torch.nn.functional.interpolate(
                    f.unsqueeze(0), size=(max(h, size), max(w, size)),
                    mode="bilinear", align_corners=False)[0]
                for f in frames
            ]
            _, h, w = frames[0].shape
        top = self._rand(h - size + 1)
        left = self._rand(w - size + 1)
        cropped = [f[:, top:top + size, left:left + size].contiguous() for f in frames]
        return cropped, real_len

    def sample(self, num: int, batch: int):
        """Return a batched ``(imgsList, flowList, certList)`` tuple."""
        windows = [self._load_window(num) for _ in range(batch)]
        win_frames = [w[0] for w in windows]
        real_lens = [w[1] for w in windows]
        # Stack across batch: frames[i] = (b,3,H,W).
        frames = [torch.stack([win_frames[e][i] for e in range(batch)]) for i in range(num + 1)]
        imgs, flows, certs = build_video_tuple_batched(frames, self.estimator, self.use_structure)
        # Mark padded (repeated-frame) transitions as fully occluded.
        for i in range(num):
            for e in range(batch):
                if i + 1 >= real_lens[e]:
                    certs[i][e].zero_()
        return imgs, flows, certs

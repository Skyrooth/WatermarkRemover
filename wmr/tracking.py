"""Follow a moving watermark across a video.

A watermark that hops between corners or slides around cannot use one static mask.
This module locates the marked patch in every frame by template matching on gradient
magnitude - the logo's edges stay put even as the background under a semi-transparent
overlay changes - then cleans the raw detections into a usable per-frame track.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import ffmpeg as ff
from .mask import Box

ProgressCb = Callable[[int, int], None] | None


@dataclass
class Detection:
    """Where the watermark was found in one frame, and how sure we are."""

    box: Box | None
    score: float
    confident: bool


def _features(rgb: np.ndarray) -> np.ndarray:
    """Gradient magnitude. Structure survives brightness changes; flat colour does not."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


class TemplateTracker:
    """Locate one fixed-size patch frame by frame.

    Searches a window around the previous hit first (cheap) and falls back to the whole
    frame when the local score drops - that fallback is what catches a corner hop.
    """

    def __init__(self, frame: np.ndarray, box: Box, search_radius: int = 120,
                 threshold: float = 0.45):
        self.width, self.height = box.w, box.h
        self.template = _features(frame[box.y:box.y + box.h, box.x:box.x + box.w])
        self.search_radius = search_radius
        self.threshold = threshold
        self.frame_w, self.frame_h = frame.shape[1], frame.shape[0]

    def _match(self, scene: np.ndarray, offset: tuple[int, int],
               allow_outside: bool = False) -> tuple[Box, float]:
        """Best template position in `scene`, reported in whole-frame coordinates.

        With `allow_outside` the scene is padded first, so a watermark half off the
        edge of the frame can still be located instead of being pinned to the border.
        """
        pad_x = self.width // 2 if allow_outside else 0
        pad_y = self.height // 2 if allow_outside else 0
        if pad_x or pad_y:
            scene = cv2.copyMakeBorder(scene, pad_y, pad_y, pad_x, pad_x,
                                       cv2.BORDER_REPLICATE)

        result = cv2.matchTemplate(scene, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        return (Box(offset[0] + loc[0] - pad_x, offset[1] + loc[1] - pad_y,
                    self.width, self.height), float(score))

    def locate(self, frame: np.ndarray, near: Box | None) -> Detection:
        feats = _features(frame)

        best: tuple[Box, float] | None = None
        if near is not None:
            x0 = max(0, near.x - self.search_radius)
            y0 = max(0, near.y - self.search_radius)
            x1 = min(self.frame_w, near.x + near.w + self.search_radius)
            y1 = min(self.frame_h, near.y + near.h + self.search_radius)
            if x1 - x0 >= self.width and y1 - y0 >= self.height:
                # Padded, so a watermark sliding off the frame edge stays trackable.
                best = self._match(feats[y0:y1, x0:x1], (x0, y0), allow_outside=True)

        # Local search missed (or was skipped): sweep the whole frame.
        if best is None or best[1] < self.threshold:
            full = self._match(feats, (0, 0), allow_outside=True)
            if best is None or full[1] > best[1]:
                best = full

        box, score = best
        # Boxes may stick out of frame; painting the mask clamps them later.
        return Detection(box, score, score >= self.threshold)


def _interpolate(track: list[Detection]) -> list[Box]:
    """Fill unconfident frames by interpolating between the confident ones."""
    anchors = [i for i, d in enumerate(track) if d.confident and d.box is not None]
    if not anchors:
        raise ValueError(
            "Watermark never matched confidently. Check the marked region, or lower "
            "--track-threshold."
        )

    boxes: list[Box] = [track[i].box for i in range(len(track))]  # type: ignore[misc]
    for i in range(len(track)):
        if track[i].confident and track[i].box is not None:
            continue
        before = [a for a in anchors if a < i]
        after = [a for a in anchors if a > i]
        if before and after:
            lo, hi = before[-1], after[0]
            t = (i - lo) / (hi - lo)
            a, b = track[lo].box, track[hi].box
            boxes[i] = Box(int(round(a.x + (b.x - a.x) * t)),
                           int(round(a.y + (b.y - a.y) * t)), a.w, a.h)
        else:  # before the first or after the last anchor: hold the nearest one
            nearest = before[-1] if before else after[0]
            boxes[i] = track[nearest].box  # type: ignore[assignment]
    return boxes


def _smooth(boxes: list[Box], window: int = 3) -> list[Box]:
    """Reject outliers using a local median - but keep positions that are merely fast.

    Replacing every position with its local median also flattens real direction
    changes: for a bouncing logo the median of [8, 1, 8] is 8, so the turning point
    disappears and the mask lags the watermark. So the median only overrides a frame
    that disagrees with its neighbours by more than the clip's own motion scale.
    """
    if window < 3 or len(boxes) < window:
        return boxes

    half = window // 2
    xs = np.array([b.x for b in boxes], dtype=np.float64)
    ys = np.array([b.y for b in boxes], dtype=np.float64)
    steps = np.hypot(np.diff(xs), np.diff(ys))
    tolerance = max(4.0, 3.0 * float(np.median(steps)) if steps.size else 4.0)

    out: list[Box] = []
    for i, box in enumerate(boxes):
        lo, hi = max(0, i - half), min(len(boxes), i + half + 1)
        mx, my = float(np.median(xs[lo:hi])), float(np.median(ys[lo:hi]))
        if np.hypot(box.x - mx, box.y - my) > tolerance:
            out.append(Box(int(round(mx)), int(round(my)), box.w, box.h))
        else:
            out.append(box)
    return out


def track_watermark(src: str | Path, box: Box, *, search_radius: int = 120,
                    threshold: float = 0.45, smooth_window: int = 3,
                    reference_frame: np.ndarray | None = None,
                    progress: ProgressCb = None) -> list[Box]:
    """Return one box per frame. Pass 1 of the two-pass moving-watermark pipeline.

    `box` marks the watermark in `reference_frame` (the first frame if not given).
    """
    info = ff.probe(src)
    reader = ff.open_reader(src)
    frame_bytes = info.width * info.height * 3

    first = reference_frame
    tracker: TemplateTracker | None = None
    if first is not None:
        tracker = TemplateTracker(first, box, search_radius, threshold)

    detections: list[Detection] = []
    previous: Box | None = None
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
            if tracker is None:
                tracker = TemplateTracker(frame, box, search_radius, threshold)

            found = tracker.locate(frame, previous)
            detections.append(found)
            if found.confident:
                previous = found.box
            if progress and (len(detections) % 5 == 0 or len(detections) == info.n_frames):
                progress(len(detections), info.n_frames)
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()

    if not detections:
        raise ValueError(f"No frames decoded from {src}")
    if progress:
        progress(len(detections), len(detections))

    return _smooth(_interpolate(detections), smooth_window)


def track_quality(track: list[Box]) -> dict:
    """Summary stats, handy for logging and for the UI."""
    xs = np.array([b.x for b in track])
    ys = np.array([b.y for b in track])
    steps = np.hypot(np.diff(xs), np.diff(ys)) if len(track) > 1 else np.array([0.0])
    return {
        "frames": len(track),
        "x_range": [int(xs.min()), int(xs.max())],
        "y_range": [int(ys.min()), int(ys.max())],
        "max_step_px": float(steps.max()),
        "moved": bool(np.ptp(xs) > 2 or np.ptp(ys) > 2),
    }

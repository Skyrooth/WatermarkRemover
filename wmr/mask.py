"""Mask construction: boxes (absolute or percent), mask images, dilation, bboxes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def clamped(self, width: int, height: int) -> "Box":
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return Box(x, y, w, h)

    def as_ffmpeg(self) -> str:
        return f"x={self.x}:y={self.y}:w={self.w}:h={self.h}"


def parse_box(spec: str, width: int, height: int) -> Box:
    """Parse 'x,y,w,h'. Each field may be a percentage: '70%,85%,28%,12%'."""
    parts = [p.strip() for p in spec.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"Box must be 'x,y,w,h', got {spec!r}")
    refs = (width, height, width, height)
    values: list[int] = []
    for part, ref in zip(parts, refs):
        if part.endswith("%"):
            values.append(int(round(float(part[:-1]) / 100.0 * ref)))
        else:
            values.append(int(round(float(part))))
    return Box(*values).clamped(width, height)


def mask_from_boxes(shape: tuple[int, int], boxes: list[Box]) -> np.ndarray:
    """White (255) inside the boxes, black elsewhere. shape is (height, width)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for box in boxes:
        mask[box.y:box.y + box.h, box.x:box.x + box.w] = 255
    return mask


def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Read a mask image; any non-black pixel counts as 'remove this'."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Cannot read mask image: {path}")
    if raw.ndim == 3:
        raw = raw[:, :, 3] if raw.shape[2] == 4 else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(raw > 8, 255, 0).astype(np.uint8)


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow the mask so inpainting never samples leftover watermark edge pixels."""
    if pixels <= 0:
        return mask
    size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel)


def bbox(mask: np.ndarray) -> Box | None:
    """Tight bounding box of the non-zero mask area."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return Box(int(xs.min()), int(ys.min()),
               int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def boxes_from_mask(mask: np.ndarray, min_area: int = 16) -> list[Box]:
    """Connected components of a mask as boxes (used by the ffmpeg delogo backend)."""
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append(Box(int(x), int(y), int(w), int(h)))
    return boxes

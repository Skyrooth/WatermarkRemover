"""Locate a constant semi-transparent watermark from the video itself.

A watermark that is present in every frame leaves a floor the background cannot go
below: take the per-pixel minimum over time and the content darkens away while the
overlay stays. That gives the position and extent without anyone eyeballing a corner.

Usage: uv run python scripts/find_watermark.py <video> [--out samples/wm_probe.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmr import ffmpeg as ff  # noqa: E402
from wmr.mask import Box  # noqa: E402


def temporal_stats(path: Path, max_frames: int = 400) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-pixel minimum and maximum brightness over the clip."""
    info = ff.probe(path)
    reader = ff.open_reader(path)
    frame_bytes = info.width * info.height * 3

    minimum = np.full((info.height, info.width), 255, np.uint8)
    maximum = np.zeros((info.height, info.width), np.uint8)
    count = 0
    try:
        while count < max_frames:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            np.minimum(minimum, gray, out=minimum)
            np.maximum(maximum, gray, out=maximum)
            count += 1
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()
    return minimum, maximum, count


def find_box(minimum: np.ndarray, percentile: float = 99.5) -> tuple[Box | None, np.ndarray]:
    """Brightest connected blob in the temporal minimum."""
    blurred = cv2.GaussianBlur(minimum, (0, 0), 1.2)
    threshold = float(np.percentile(blurred, percentile))
    mask = (blurred >= max(threshold, 12)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, 0
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area > best_area:
            best, best_area = Box(int(x), int(y), int(w), int(h)), int(area)
    return best, mask * 255


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("samples/wm_probe.png"))
    parser.add_argument("--percentile", type=float, default=99.5)
    args = parser.parse_args(argv)

    minimum, maximum, frames = temporal_stats(args.video)
    box, mask = find_box(minimum, args.percentile)
    info = ff.probe(args.video)

    print(f"{args.video.name}: {info.width}x{info.height}, {frames} frames scanned")
    if box is None:
        print("no constant bright region found")
        return 1

    pad = 30
    x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
    x1 = min(info.width, box.x + box.w + pad)
    y1 = min(info.height, box.y + box.h + pad)

    print(f"watermark box : x={box.x} y={box.y} w={box.w} h={box.h}")
    print(f"  as percent  : {100*box.x/info.width:.1f}%,{100*box.y/info.height:.1f}%,"
          f"{100*box.w/info.width:.1f}%,{100*box.h/info.height:.1f}%")
    print(f"  from right  : {info.width - (box.x + box.w)} px")
    print(f"  from bottom : {info.height - (box.y + box.h)} px")
    print(f"  min inside  : {int(minimum[box.y:box.y+box.h, box.x:box.x+box.w].max())} "
          f"(brightest floor)  outside: {int(np.median(minimum))}")
    print(f"  CLI         : --box {box.x},{box.y},{box.w},{box.h}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel = np.hstack([
        cv2.cvtColor(minimum[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(maximum[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
    ])
    cv2.imwrite(str(args.out), panel)
    print(f"wrote {args.out}  (temporal min | detected mask | temporal max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

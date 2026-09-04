"""Score watermark removal against the clean reference clip.

For each frame we compare the cleaned video with the watermark-free twin, restricted to
the region the watermark occupied. "Residue" is the mean absolute error there, on a
0-255 scale; "before" is the same measure on the untouched input, so the ratio says how
much of the watermark actually went away.

Usage:
  uv run python scripts/eval_removal.py                 # default sweep
  uv run python scripts/eval_removal.py migan:6 telea:3 # engine:grow pairs
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmr import ffmpeg as ff  # noqa: E402
from wmr.mask import Box, mask_from_boxes  # noqa: E402
from wmr.tracking import track_watermark  # noqa: E402
from wmr.video_ops import remove_from_video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
TRUTH = json.loads((SAMPLES / "truth.json").read_text(encoding="utf-8"))
BOXES = [Box(*b) for b in TRUTH["moving_boxes"]]
STATIC = Box(*TRUTH["static_box"])


def _frames(path: Path):
    info = ff.probe(path)
    reader = ff.open_reader(path)
    n = info.width * info.height * 3
    try:
        while True:
            raw = reader.stdout.read(n)
            if len(raw) < n:
                return
            yield np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3)
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()


def residue(candidate: Path, reference: Path, boxes: list[Box], pad: int = 0) -> float:
    """Mean absolute error inside the watermark region, over every frame."""
    total, count = 0.0, 0
    for i, (got, want) in enumerate(zip(_frames(candidate), _frames(reference))):
        box = boxes[min(i, len(boxes) - 1)]
        x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
        x1 = min(got.shape[1], box.x + box.w + pad)
        y1 = min(got.shape[0], box.y + box.h + pad)
        diff = np.abs(got[y0:y1, x0:x1].astype(np.int16) - want[y0:y1, x0:x1].astype(np.int16))
        total += float(diff.mean())
        count += 1
    return total / max(count, 1)


def main(argv: list[str]) -> int:
    combos = argv or ["delogo:3", "telea:3", "migan:3", "migan:6", "migan:10", "migan:16"]

    info = ff.probe(SAMPLES / "wm_moving.mp4")
    shape = (info.height, info.width)

    base_moving = residue(SAMPLES / "wm_moving.mp4", SAMPLES / "ref_moving.mp4", BOXES)
    base_static = residue(SAMPLES / "wm_static.mp4", SAMPLES / "ref_static.mp4", [STATIC])
    print(f"untouched input   moving {base_moving:6.2f}   static {base_static:6.2f}\n")

    seed_mask = mask_from_boxes(shape, [BOXES[0]])
    for clip, ref, label in (("wm_moving.mp4", "ref_moving.mp4", "MOVING, smooth background"),
                             ("wm_textured.mp4", "ref_textured.mp4", "MOVING, textured background")):
        base = residue(SAMPLES / clip, SAMPLES / ref, BOXES)
        print(f"{label} (--track), untouched {base:.2f}")
        track = track_watermark(SAMPLES / clip, BOXES[0],
                                reference_frame=ff.grab_frame(SAMPLES / clip, 0.0))
        for combo in combos:
            engine, _, grow = combo.partition(":")
            grow = int(grow or 3)
            dst = SAMPLES / f"eval_{engine}_{grow}.mp4"
            started = time.time()
            try:
                remove_from_video(SAMPLES / clip, dst, seed_mask, engine=engine,
                                  grow=grow, track=track)
            except ValueError as exc:
                print(f"  {engine:7s} grow {grow:2d}   skipped: "
                      f"{str(exc).splitlines()[0][:55]}")
                continue
            elapsed = time.time() - started
            err = residue(dst, SAMPLES / ref, BOXES)
            print(f"  {engine:7s} grow {grow:2d}   residue {err:6.2f}  "
                  f"({100 * err / base:5.1f}% of original)  {elapsed:5.1f}s")
            dst.unlink(missing_ok=True)
        print()

    print("\nSTATIC clip (no tracking)")
    static_mask = mask_from_boxes(shape, [STATIC])
    for combo in combos:
        engine, _, grow = combo.partition(":")
        grow = int(grow or 3)
        dst = SAMPLES / f"eval_static_{engine}_{grow}.mp4"
        started = time.time()
        remove_from_video(SAMPLES / "wm_static.mp4", dst, static_mask,
                          engine=engine, grow=grow)
        elapsed = time.time() - started
        err = residue(dst, SAMPLES / "ref_static.mp4", [STATIC])
        print(f"  {engine:7s} grow {grow:2d}   residue {err:6.2f}  "
              f"({100 * err / base_static:5.1f}% of original)  {elapsed:5.1f}s")
        dst.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Measure the tracker against known ground truth.

samples/truth.json records the exact box of every frame written by make_samples.py,
so tracking error is a real number in pixels rather than an impression.

Usage: uv run python scripts/eval_tracking.py [clip]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmr import ffmpeg as ff  # noqa: E402
from wmr.mask import Box  # noqa: E402
from wmr.tracking import track_quality, track_watermark  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def evaluate(clip: Path) -> float:
    truth = json.loads((SAMPLES / "truth.json").read_text(encoding="utf-8"))
    boxes = [Box(*b) for b in truth["moving_boxes"]]

    first = ff.grab_frame(clip, 0.0)
    track = track_watermark(clip, boxes[0], reference_frame=first)
    print(f"{clip.name}: {track_quality(track)}")

    errors = np.array([float(np.hypot(got.x - want.x, got.y - want.y))
                       for got, want in zip(track, boxes)])
    print(f"  frames {len(errors)}  mean {errors.mean():5.2f}px  "
          f"median {np.median(errors):5.2f}px  p95 {np.percentile(errors, 95):6.2f}px  "
          f"max {errors.max():6.2f}px")
    print(f"  within 2px {100 * (errors <= 2).mean():5.1f}%   "
          f"5px {100 * (errors <= 5).mean():5.1f}%   "
          f"15px {100 * (errors <= 15).mean():5.1f}%")
    for i in np.argsort(errors)[-3:][::-1]:
        print(f"    worst frame {int(i):4d}: got ({track[i].x:4d},{track[i].y:4d}) "
              f"want ({boxes[i].x:4d},{boxes[i].y:4d})  err {errors[i]:6.1f}px")
    return float(errors.mean())


def main(argv: list[str]) -> int:
    clips = [SAMPLES / name for name in (argv or ["wm_moving.mp4", "wm_textured.mp4"])]
    for clip in clips:
        if not clip.exists():
            print(f"missing {clip} - run: uv run python scripts/make_samples.py")
            return 1
        evaluate(clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Benchmark every inpainting engine on the sample frame.

Usage: uv run python scripts/bench_lama.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmr.engines import ENGINE_NAMES, get_engine  # noqa: E402
from wmr.image_ops import read_image, write_image  # noqa: E402
from wmr.mask import dilate, mask_from_boxes, parse_box  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "wm_image.png"
BOX = "940,590,300,90"
RUNS = 3


def main() -> int:
    rgb, _ = read_image(SAMPLE)
    height, width = rgb.shape[:2]
    mask = dilate(mask_from_boxes((height, width), [parse_box(BOX, width, height)]), 3)

    for name in ENGINE_NAMES:
        try:
            started = time.time()
            engine = get_engine(name)
            load = time.time() - started

            engine.inpaint(rgb, mask)  # warm-up: first run pays for graph compilation
            started = time.time()
            for _ in range(RUNS):
                out = engine.inpaint(rgb, mask)
            per_frame = (time.time() - started) / RUNS

            provider = getattr(engine, "providers", ["cpu (opencv)"])[0]
            fallback = getattr(engine, "fallback_from", None)
            note = f"  (fell back from {fallback})" if fallback else ""
            print(f"{name:6s} {provider:24s} load {load:5.2f}s  "
                  f"{per_frame:6.3f}s/frame{note}")
            write_image(SAMPLE.parent / f"img_{name}.png", out)
        except Exception as exc:  # noqa: BLE001 - report and move on to the next engine
            print(f"{name:6s} FAILED: {str(exc).splitlines()[0][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

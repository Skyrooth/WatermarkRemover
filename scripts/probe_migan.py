"""Probe the MI-GAN ONNX model: mask polarity, accepted sizes, provider support.

Usage: uv run python scripts/probe_migan.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = Path(__file__).resolve().parent.parent / "models" / "migan.onnx"


def scene(size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """A textured background with a bright block standing in for a watermark."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.dstack([
        (128 + 100 * np.sin(xx / 24)),
        (128 + 100 * np.sin(yy / 31)),
        (128 + 100 * np.sin((xx + yy) / 40)),
    ]).clip(0, 255).astype(np.uint8)
    clean = img.copy()
    img[300:380, 150:400] = (255, 255, 255)
    hole = np.zeros((size, size), np.uint8)
    hole[300:380, 150:400] = 255
    return img, hole, clean


def run(session, image: np.ndarray, hole: np.ndarray, invert: bool) -> np.ndarray:
    mask = (255 - hole) if invert else hole
    feeds = {
        "image": image.transpose(2, 0, 1)[None].astype(np.uint8),
        "mask": mask[None, None].astype(np.uint8),
    }
    return session.run(None, feeds)[0][0].transpose(1, 2, 0)


def main() -> int:
    image, hole, clean = scene()

    for provider in ("DmlExecutionProvider", "CPUExecutionProvider"):
        if provider not in ort.get_available_providers():
            continue
        try:
            started = time.time()
            session = ort.InferenceSession(str(MODEL), providers=[provider])
            load = time.time() - started
            out = run(session, image, hole, invert=False)
            started = time.time()
            for _ in range(3):
                out = run(session, image, hole, invert=False)
            print(f"{provider:24s} OK   load {load:5.2f}s  {(time.time() - started) / 3:5.3f}s/frame")
        except Exception as exc:  # noqa: BLE001
            print(f"{provider:24s} FAIL {str(exc)[:120]}")
            continue

        # Which polarity actually repairs the hole?
        for invert in (False, True):
            out = run(session, image, hole, invert=invert)
            err = float(np.abs(out[300:380, 150:400].astype(int)
                               - clean[300:380, 150:400].astype(int)).mean())
            kept = float(np.abs(out[:200].astype(int) - image[:200].astype(int)).mean())
            label = "mask 255=keep" if invert else "mask 255=hole"
            print(f"    {label:16s} hole-vs-truth {err:6.1f}   outside-drift {kept:5.1f}")
            cv2.imwrite(str(MODEL.parent.parent / "samples" /
                            f"migan_probe_{'keep' if invert else 'hole'}.png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    # Accepted input sizes (the graph says dynamic, the network may disagree).
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    for size in (128, 256, 384, 512, 640, 768, 1024):
        img = cv2.resize(image, (size, size))
        msk = cv2.resize(hole, (size, size), interpolation=cv2.INTER_NEAREST)
        try:
            out = run(session, img, msk, invert=False)
            print(f"    size {size:4d} -> {out.shape}")
        except Exception as exc:  # noqa: BLE001
            print(f"    size {size:4d} -> FAIL {str(exc)[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

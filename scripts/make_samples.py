"""Generate the test corpus deterministically.

ffmpeg's synthetic sources are either random per run (`gradients`) or adversarial for
inpainting (`testsrc2`), and neither gives a watermark-free twin to score against. So
the frames are built in numpy instead: for every clip we write both the watermarked
video and a pixel-identical clean reference, plus the exact logo position per frame.
That turns "looks clean" into a number.

Usage: uv run python scripts/make_samples.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmr.ffmpeg import FFMPEG, require_ffmpeg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

WIDTH, HEIGHT, FPS = 1280, 720, 30
STATIC_BOX = (940, 590)
SPEED_X, SPEED_Y = 260.0, 170.0


def background(frame_index: int) -> np.ndarray:
    """A smooth, slowly drifting colour field - deterministic and easy to score."""
    t = frame_index / FPS
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    r = 128 + 110 * np.sin(xx / 70.0 + t * 1.1)
    g = 128 + 110 * np.sin(yy / 55.0 - t * 0.7)
    b = 128 + 110 * np.sin((xx + yy) / 90.0 + t * 0.4)
    return np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)


def _octave(seed: int, cells: int) -> np.ndarray:
    """One layer of value noise: a small random grid stretched up smoothly."""
    rng = np.random.default_rng(seed)
    grid = rng.random((cells, cells), dtype=np.float32)
    return cv2.resize(grid, (WIDTH + 256, HEIGHT + 256), interpolation=cv2.INTER_CUBIC)


def textured_background(frame_index: int, _cache: dict = {}) -> np.ndarray:
    """Fractal value noise - statistics much closer to real footage than a sine field.

    A smooth gradient flatters diffusion inpainting and LaMa's Fourier convolutions,
    which is exactly the kind of bias that picks the wrong default. This clip is the
    counterweight: broadband detail with no periodic structure to exploit.
    """
    if "octaves" not in _cache:
        # Seven octaves down to ~3px cells, rolling off gently (1/1.7^i) so the fine
        # detail actually survives instead of being drowned by the base layer.
        _cache["octaves"] = [(_octave(11 + i, 8 << i), 1.0 / (1.7 ** i), 2 + 3 * i)
                             for i in range(7)]

    t = frame_index / FPS
    field = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    for layer, weight, drift in _cache["octaves"]:
        dx = int(128 + drift * t * 6) % 256
        dy = int(128 + drift * t * 4) % 256
        window = layer[dy:dy + HEIGHT, dx:dx + WIDTH]
        for channel in range(3):
            field[:, :, channel] += weight * np.roll(window, channel * 37, axis=1)

    field -= field.min()
    field /= max(field.max(), 1e-6)
    return (30 + 200 * field).clip(0, 255).astype(np.uint8)


def logo_rgba() -> np.ndarray:
    return np.array(Image.open(SAMPLES / "logo.png").convert("RGBA"))


def moving_position(frame_index: int, logo_w: int, logo_h: int) -> tuple[int, int]:
    """Triangle-wave bounce, the same shape an ffmpeg overlay expression would give."""
    t = frame_index / FPS
    span_x, span_y = WIDTH - logo_w, HEIGHT - logo_h
    x = abs((t * SPEED_X) % (2 * span_x) - span_x)
    y = abs((t * SPEED_Y) % (2 * span_y) - span_y)
    return int(round(x)), int(round(y))


def composite(frame: np.ndarray, logo: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = logo.shape[:2]
    x = max(0, min(x, WIDTH - w))
    y = max(0, min(y, HEIGHT - h))
    patch = frame[y:y + h, x:x + w].astype(np.float32)
    alpha = (logo[:, :, 3:4].astype(np.float32) / 255.0)
    blended = logo[:, :, :3].astype(np.float32) * alpha + patch * (1 - alpha)
    out = frame.copy()
    out[y:y + h, x:x + w] = blended.clip(0, 255).astype(np.uint8)
    return out


def encode(path: Path, frames, with_audio: bool = True) -> None:
    """Lossless x264 so the samples themselves add no error to any measurement."""
    require_ffmpeg()
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=330:duration=6",
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", str(path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"encode failed for {path}:\n{stderr.strip()}")


def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    logo = logo_rgba()
    lh, lw = logo.shape[:2]
    n_frames = 6 * FPS

    print("building frames...")
    clean = [background(i) for i in range(n_frames)]

    static = [composite(f, logo, *STATIC_BOX) for f in clean]
    positions = [moving_position(i, lw, lh) for i in range(n_frames)]
    moving = [composite(f, logo, *pos) for f, pos in zip(clean, positions)]

    print("building textured frames...")
    rough = [textured_background(i) for i in range(n_frames)]
    rough_moving = [composite(f, logo, *pos) for f, pos in zip(rough, positions)]

    print("encoding...")
    encode(SAMPLES / "ref_static.mp4", clean)
    encode(SAMPLES / "wm_static.mp4", static)
    encode(SAMPLES / "ref_moving.mp4", clean)
    encode(SAMPLES / "wm_moving.mp4", moving)
    encode(SAMPLES / "ref_textured.mp4", rough)
    encode(SAMPLES / "wm_textured.mp4", rough_moving)

    meta = {
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": n_frames,
        "logo": {"w": lw, "h": lh},
        "static_box": [STATIC_BOX[0], STATIC_BOX[1], lw, lh],
        "moving_boxes": [[x, y, lw, lh] for x, y in positions],
    }
    (SAMPLES / "truth.json").write_text(json.dumps(meta), encoding="utf-8")

    # Keep a single watermarked still around for the image tests and benchmarks.
    Image.fromarray(static[60]).save(SAMPLES / "wm_image.png")
    Image.fromarray(clean[60]).save(SAMPLES / "ref_image.png")
    Image.fromarray(composite(rough[60], logo, *STATIC_BOX)).save(
        SAMPLES / "wm_image_textured.png")
    Image.fromarray(rough[60]).save(SAMPLES / "ref_image_textured.png")

    print(f"wrote {n_frames} frames x 4 clips + truth.json to {SAMPLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

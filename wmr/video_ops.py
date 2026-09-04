"""Video watermark removal: a fast ffmpeg backend and a per-frame inpainting backend."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from . import ffmpeg as ff
from .engines import Engine, get_engine
from .mask import Box, boxes_from_mask, dilate as dilate_mask

ProgressCb = Callable[[int, int], None] | None


def _delogo_filter(boxes: list[Box], width: int, height: int) -> str:
    """delogo refuses boxes touching the frame border, so keep 1px of margin."""
    parts = []
    for box in boxes:
        x = max(1, box.x)
        y = max(1, box.y)
        w = max(1, min(box.w, width - x - 1))
        h = max(1, min(box.h, height - y - 1))
        parts.append(f"delogo=x={x}:y={y}:w={w}:h={h}")
    return ",".join(parts)


def remove_with_delogo(src: str | Path, dst: str | Path, mask: np.ndarray,
                       crf: int = 18, preset: str = "medium") -> dict:
    """ffmpeg's delogo filter: interpolates each box from its border. Real-time speed."""
    info = ff.probe(src)
    boxes = boxes_from_mask(mask)
    if not boxes:
        raise ValueError("Mask is empty - nothing to remove")

    chain = _delogo_filter(boxes, info.width, info.height)
    cmd = [ff.FFMPEG, "-y", "-v", "error", "-i", str(src), "-vf", chain,
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += ["-c:a", "copy"] if info.has_audio else ["-an"]
    cmd += [str(dst)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg delogo failed:\n{proc.stderr.strip()}")

    return {"engine": "delogo", "boxes": len(boxes), "frames": info.n_frames,
            "audio": info.has_audio, "output": str(dst)}


def remove_per_frame(src: str | Path, dst: str | Path, mask: np.ndarray,
                     engine: str | Engine = "telea", crf: int = 18,
                     preset: str = "medium", progress: ProgressCb = None,
                     **engine_kwargs) -> dict:
    """Decode -> inpaint every frame -> re-encode, with the original audio copied over."""
    info = ff.probe(src)
    if mask.shape != (info.height, info.width):
        raise ValueError(f"Mask {mask.shape} does not match video {(info.height, info.width)}")

    eng = engine if isinstance(engine, Engine) else get_engine(engine, **engine_kwargs)
    frame_bytes = info.width * info.height * 3

    reader = ff.open_reader(src)
    writer = ff.open_writer(dst, src, info, crf=crf, preset=preset)
    count = 0
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
            writer.stdin.write(eng.inpaint(frame, mask).tobytes())
            count += 1
            if progress and (count % 5 == 0 or count == info.n_frames):
                progress(count, info.n_frames)
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()
        if writer.stdin:
            writer.stdin.close()
        stderr = writer.stderr.read().decode(errors="replace") if writer.stderr else ""
        code = writer.wait()

    if code != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{stderr.strip()}")
    if progress:
        progress(count, count)

    return {"engine": eng.name, "frames": count, "audio": info.has_audio,
            "size": (info.width, info.height), "output": str(dst)}


def _expand(box: Box, pixels: int, width: int, height: int) -> Box:
    """Grow a box on every side. For a rectangle this is exactly a mask dilation."""
    return Box(box.x - pixels, box.y - pixels,
               box.w + 2 * pixels, box.h + 2 * pixels).clamped(width, height)


def remove_tracked(src: str | Path, dst: str | Path, track: list[Box],
                   engine: str | Engine = "migan", grow: int = 3, crf: int = 18,
                   preset: str = "medium", progress: ProgressCb = None,
                   **engine_kwargs) -> dict:
    """Pass 2 of the moving-watermark pipeline: one mask position per frame."""
    if not track:
        raise ValueError("Empty track - nothing to remove")

    info = ff.probe(src)
    eng = engine if isinstance(engine, Engine) else get_engine(engine, **engine_kwargs)
    frame_bytes = info.width * info.height * 3

    # One reusable buffer: paint the box, inpaint, wipe it again. Allocating a fresh
    # full-frame mask per frame would cost more than the inpainting on long clips.
    mask = np.zeros((info.height, info.width), np.uint8)
    boxes = [_expand(b, grow, info.width, info.height) for b in track]

    reader = ff.open_reader(src)
    writer = ff.open_writer(dst, src, info, crf=crf, preset=preset)
    count = 0
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
            box = boxes[count] if count < len(boxes) else boxes[-1]
            mask[box.y:box.y + box.h, box.x:box.x + box.w] = 255
            writer.stdin.write(eng.inpaint(frame, mask).tobytes())
            mask[box.y:box.y + box.h, box.x:box.x + box.w] = 0
            count += 1
            if progress and (count % 5 == 0 or count == info.n_frames):
                progress(count, info.n_frames)
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()
        if writer.stdin:
            writer.stdin.close()
        stderr = writer.stderr.read().decode(errors="replace") if writer.stderr else ""
        code = writer.wait()

    if code != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{stderr.strip()}")
    if progress:
        progress(count, count)

    return {"engine": eng.name, "frames": count, "audio": info.has_audio,
            "size": (info.width, info.height), "tracked": True,
            "track_frames": len(track), "output": str(dst)}


def remove_from_video(src: str | Path, dst: str | Path, mask: np.ndarray,
                      engine: str = "delogo", grow: int = 3, crf: int = 18,
                      preset: str = "medium", progress: ProgressCb = None,
                      track: list[Box] | None = None, **engine_kwargs) -> dict:
    if track is not None:
        if engine == "delogo":
            raise ValueError(
                "delogo draws one fixed box for the whole clip, so it cannot follow a "
                "moving watermark. Use --engine migan (or telea/ns/lama) with --track."
            )
        return remove_tracked(src, dst, track, engine=engine, grow=grow, crf=crf,
                              preset=preset, progress=progress, **engine_kwargs)

    work_mask = dilate_mask(mask, grow)
    if engine == "delogo":
        return remove_with_delogo(src, dst, work_mask, crf=crf, preset=preset)
    return remove_per_frame(src, dst, work_mask, engine=engine, crf=crf,
                            preset=preset, progress=progress, **engine_kwargs)

"""Thin ffmpeg/ffprobe wrappers: probing, raw frame pipes, audio-preserving mux."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class FfmpegMissing(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if not FFMPEG or not FFPROBE:
        raise FfmpegMissing(
            "ffmpeg/ffprobe not found on PATH. Install a full build "
            "(https://www.gyan.dev/ffmpeg/builds/) and reopen the terminal."
        )


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    has_audio: bool
    codec: str

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def _parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata. Frame count falls back to fps*duration when absent."""
    require_ffmpeg()
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or 30.0
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)

    n_frames = 0
    for key in ("nb_frames", "nb_read_frames"):
        try:
            n_frames = int(video.get(key) or 0)
        except (TypeError, ValueError):
            n_frames = 0
        if n_frames:
            break
    if not n_frames and duration and fps:
        n_frames = int(round(duration * fps))

    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        n_frames=n_frames,
        duration=duration,
        has_audio=has_audio,
        codec=str(video.get("codec_name", "")),
    )


def grab_frame(path: str | Path, at_seconds: float = 0.0):
    """Decode a single RGB frame (numpy uint8 HxWx3) for previews and mask drawing."""
    import numpy as np

    require_ffmpeg()
    info = probe(path)
    cmd = [FFMPEG, "-v", "error"]
    if at_seconds > 0:
        cmd += ["-ss", f"{at_seconds:.3f}"]
    cmd += ["-i", str(path), "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    expected = info.width * info.height * 3
    if len(raw) < expected:
        raise RuntimeError(f"Short frame read ({len(raw)}/{expected} bytes) from {path}")
    return np.frombuffer(raw[:expected], dtype="uint8").reshape(info.height, info.width, 3).copy()


def open_reader(path: str | Path) -> subprocess.Popen:
    """rawvideo rgb24 decoder pipe."""
    require_ffmpeg()
    return subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8,
    )


def open_writer(dest: str | Path, source: str | Path, info: VideoInfo,
                crf: int = 18, preset: str = "medium") -> subprocess.Popen:
    """Encoder pipe that maps audio straight from the source file (stream copy)."""
    require_ffmpeg()
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{info.width}x{info.height}", "-r", f"{info.fps:.6f}",
        "-i", "pipe:0",
        "-i", str(source),
        "-map", "0:v:0",
    ]
    if info.has_audio:
        cmd += ["-map", "1:a:0?", "-c:a", "copy"]
    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-shortest", str(dest),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=10 ** 8)

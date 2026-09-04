"""Model downloads for the ONNX engines."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .engines import LAMA_PATH, MIGAN_PATH

MODELS: dict[str, tuple[str, Path, str]] = {
    "migan": (
        "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan.onnx",
        MIGAN_PATH,
        "MI-GAN pipeline, ~30 MB, runs on the GPU via DirectML",
    ),
    "lama": (
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        LAMA_PATH,
        "LaMa fp32, ~208 MB, CPU only (DirectML cannot run its FFT ops)",
    ),
}


def download_model(name: str) -> Path:
    """Fetch one model into models/ and return its path. Existing files are kept."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {', '.join(MODELS)}")
    url, dest, note = MODELS[name]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{name}: already present at {dest}")
        return dest

    print(f"{name} - {note}\n  {url}\n  -> {dest}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
    print()
    tmp.replace(dest)
    return dest

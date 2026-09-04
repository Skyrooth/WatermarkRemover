"""Image watermark removal."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .engines import Engine, get_engine
from .mask import dilate as dilate_mask


def read_image(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (rgb, alpha). Alpha is kept aside so transparency survives a round trip."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Cannot read image: {path}")
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB), None
    if raw.shape[2] == 4:
        return cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB), raw[:, :, 3]
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), None


def write_image(path: str | Path, rgb: np.ndarray, alpha: np.ndarray | None = None) -> None:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if alpha is not None:
        out = np.dstack([out, alpha])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise RuntimeError(f"Failed to write image: {path}")


def remove_from_image(src: str | Path, dst: str | Path, mask: np.ndarray,
                      engine: str | Engine = "telea", grow: int = 3,
                      **engine_kwargs) -> dict:
    """Inpaint the masked region of one image and write the result."""
    rgb, alpha = read_image(src)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"Mask {mask.shape} does not match image {rgb.shape[:2]}")

    work_mask = dilate_mask(mask, grow)
    eng = engine if isinstance(engine, Engine) else get_engine(engine, **engine_kwargs)
    result = eng.inpaint(rgb, work_mask)
    write_image(dst, result, alpha)

    return {
        "engine": eng.name,
        "size": (rgb.shape[1], rgb.shape[0]),
        "masked_pixels": int((work_mask > 0).sum()),
        "output": str(dst),
    }

"""Inpainting engines. OpenCV backends work out of the box; the ONNX ones need a model."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np


@contextlib.contextmanager
def _muted_stderr():
    """Silence stderr at the file-descriptor level.

    onnxruntime logs from C++ straight to fd 2, so Python-level logger settings and
    `contextlib.redirect_stderr` do not touch it.
    """
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:  # no usable stderr (pythonw, redirected pipes) - nothing to mute
        yield
        return

    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LAMA_PATH = MODELS_DIR / "lama_fp32.onnx"
MIGAN_PATH = MODELS_DIR / "migan.onnx"

ENGINE_NAMES = ("telea", "ns", "migan", "lama")

# Preferred execution providers, best first. A provider that loads the graph can still
# fail at run time (DirectML has no kernel for LaMa's FFT ops), so every ONNX engine
# proves its provider with one forward pass and falls back to the CPU if it throws.
PROVIDER_PREFERENCE = (
    "CUDAExecutionProvider",   # NVIDIA, needs the CUDA toolkit + cuDNN
    "DmlExecutionProvider",    # any DX12 GPU on Windows, no toolkit needed
    "CPUExecutionProvider",
)


class Engine:
    name = "base"

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Repair the white area of `mask` (255 = remove) in an RGB uint8 image."""
        raise NotImplementedError


class OpenCVEngine(Engine):
    """Classic diffusion inpainting. Fast, no model, good for small or flat regions."""

    def __init__(self, method: str = "telea", radius: int = 5):
        self.name = method
        self.radius = radius
        self.flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, mask, self.radius, self.flag)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _crop_window(mask: np.ndarray, margin_ratio: float = 0.75,
                 min_margin: int = 64) -> tuple[int, int, int, int]:
    """Region around the mask to feed the model - context matters, whole frames do not."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mx = max(min_margin, int((x1 - x0) * margin_ratio))
    my = max(min_margin, int((y1 - y0) * margin_ratio))
    h, w = mask.shape
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx)
    y1 = min(h, y1 + my)
    return x0, y0, x1 - x0, y1 - y0


def _match_aspect(window: tuple[int, int, int, int], aspect: float,
                  frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Grow a crop window to the model's aspect ratio so nothing is squashed."""
    x, y, w, h = window
    if h * aspect > w:
        target = min(int(round(h * aspect)), frame_w)
        x = max(0, min(x - (target - w) // 2, frame_w - target))
        w = target
    else:
        target = min(int(round(w / aspect)), frame_h)
        y = max(0, min(y - (target - h) // 2, frame_h - target))
        h = target
    return x, y, w, h


class OnnxInpaintEngine(Engine):
    """Shared plumbing for ONNX inpainters: provider choice, crop, resize, seam blend.

    Subclasses only describe how their model wants the tensors (`_feed` / `_decode`).
    """

    name = "onnx"
    model_path: Path = MODELS_DIR
    install_hint = ""
    fallback_from: str | None = None

    def __init__(self, model_path: str | Path | None = None, size: int = 512,
                 providers: list[str] | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"The {self.name} engine needs onnxruntime. Install it with:\n"
                "  uv add onnxruntime-directml   (GPU on Windows, no CUDA toolkit)\n"
                "  uv add onnxruntime            (CPU only)"
            ) from exc

        self.path = Path(model_path or self.model_path)
        if not self.path.exists():
            raise RuntimeError(
                f"Model not found at {self.path}\n"
                f"Download it with:  uv run wmr download-model --model {self.name}"
            )

        available = ort.get_available_providers()
        wanted = providers or [p for p in PROVIDER_PREFERENCE if p in available]
        self._open_session(ort, wanted)
        self.size = size
        if self.model_hw is None:
            self.model_hw = (size, size)

        if self.providers[0] != "CPUExecutionProvider":
            probe_h, probe_w = self.model_hw
            # The probe is expected to fail for some model/provider pairs, so mute the
            # runtime's own error spew - a red ONNX stack trace on every successful run
            # reads like a crash. `fallback_from` records what actually happened, and
            # only the probe is silenced: the real session still logs normally.
            try:
                with _muted_stderr():
                    self._run_model(np.zeros((probe_h, probe_w, 3), np.uint8),
                                    np.full((probe_h, probe_w), 255, np.uint8))
            except Exception:  # noqa: BLE001 - any runtime failure means: use the CPU
                self.fallback_from = self.providers[0]
                self._open_session(ort, ["CPUExecutionProvider"])
                if self.model_hw is None:
                    self.model_hw = (size, size)

    def _open_session(self, ort, providers: list[str]) -> None:
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.providers = self.session.get_providers()
        inputs = self.session.get_inputs()
        self.image_input = inputs[0].name
        self.mask_input = inputs[1].name if len(inputs) > 1 else "mask"

        # Some exports are frozen at one resolution; others are fully dynamic.
        shape = list(inputs[0].shape)
        dims = shape[2:4] if len(shape) == 4 else []
        self.model_hw: tuple[int, int] | None = (
            (int(dims[0]), int(dims[1]))
            if len(dims) == 2 and all(isinstance(d, int) for d in dims) else None
        )

    def _feed(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """Model inputs for an image and a hole mask (255 = repair), both at model size."""
        raise NotImplementedError

    def _decode(self, output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _run_model(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._decode(self.session.run(None, self._feed(image, mask))[0])

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return rgb

        model_h, model_w = self.model_hw
        cx, cy, cw, ch = _match_aspect(_crop_window(mask), model_w / model_h,
                                       rgb.shape[1], rgb.shape[0])
        crop = rgb[cy:cy + ch, cx:cx + cw]
        crop_mask = mask[cy:cy + ch, cx:cx + cw]

        if (cw, ch) != (model_w, model_h):
            shrinking = model_w * model_h < cw * ch
            small = cv2.resize(crop, (model_w, model_h),
                               interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)
            # Resampling smears the watermark a pixel or two past its own edge. Any of
            # that halo left outside the hole is read by the model as valid context and
            # dragged back into the fill, so grow the mask to swallow it: partial
            # coverage counts as hole, plus a couple of pixels of margin.
            grown = cv2.resize(crop_mask, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
            small_mask = np.where(grown > 0, 255, 0).astype(np.uint8)
            small_mask = cv2.dilate(small_mask,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        else:
            small, small_mask = crop, crop_mask

        out = self._run_model(small, small_mask)
        if out.shape[:2] != (ch, cw):
            out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC)

        # Feather the seam so the repaired patch does not show a hard edge.
        alpha = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (0, 0), 1.6)[..., None]
        blended = out.astype(np.float32) * alpha + crop.astype(np.float32) * (1 - alpha)

        result = rgb.copy()
        result[cy:cy + ch, cx:cx + cw] = np.clip(blended, 0, 255).astype(np.uint8)
        return result


class LamaEngine(OnnxInpaintEngine):
    """LaMa. Highest quality on large holes; its FFT ops force CPU on DirectML."""

    name = "lama"
    model_path = LAMA_PATH

    def _feed(self, image: np.ndarray, mask: np.ndarray) -> dict:
        return {
            self.image_input: image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0,
            self.mask_input: (mask > 0).astype(np.float32)[None, None],
        }

    def _decode(self, output: np.ndarray) -> np.ndarray:
        out = output[0].transpose(1, 2, 0)
        if float(out.max()) <= 1.5:
            out = out * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)


class MiganEngine(OnnxInpaintEngine):
    """MI-GAN pipeline export: uint8 in and out, dynamic size, runs on DirectML.

    Its mask polarity is the opposite of LaMa's - 255 means *keep this pixel*.
    """

    name = "migan"
    model_path = MIGAN_PATH

    def _feed(self, image: np.ndarray, mask: np.ndarray) -> dict:
        return {
            self.image_input: image.transpose(2, 0, 1)[None].astype(np.uint8),
            self.mask_input: (255 - mask)[None, None].astype(np.uint8),
        }

    def _decode(self, output: np.ndarray) -> np.ndarray:
        return output[0].transpose(1, 2, 0).astype(np.uint8)


def get_engine(name: str, **kwargs) -> Engine:
    name = name.lower()
    if name in ("telea", "ns"):
        return OpenCVEngine(name, radius=int(kwargs.get("radius", 5)))
    if name in ("lama", "migan"):
        cls = LamaEngine if name == "lama" else MiganEngine
        return cls(model_path=kwargs.get("model_path"),
                   size=int(kwargs.get("max_side", 512)),
                   providers=kwargs.get("providers"))
    raise ValueError(f"Unknown engine {name!r}. Choose from {', '.join(ENGINE_NAMES)}")

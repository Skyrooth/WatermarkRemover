"""WMR Studio - browser UI for removing watermarks from images and video.

Run with:  uv run python app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from wmr import ffmpeg as ff
from wmr.engines import LAMA_PATH, MIGAN_PATH
from wmr.image_ops import remove_from_image, write_image
from wmr.mask import bbox, dilate
from wmr.tracking import track_quality, track_watermark
from wmr.video_ops import remove_from_video

OUT_DIR = Path(tempfile.gettempdir()) / "wmr-studio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_ENGINES = ["lama", "migan", "telea", "ns"]
VIDEO_ENGINES = ["migan", "delogo", "telea", "ns", "lama"]
MODEL_PATHS = {"migan": MIGAN_PATH, "lama": LAMA_PATH}


def mask_from_editor(editor_value: dict | None) -> np.ndarray | None:
    """Turn the brush strokes of gr.ImageEditor into a binary mask."""
    if not editor_value:
        return None
    background = editor_value.get("background")
    if background is None:
        return None
    height, width = background.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for layer in editor_value.get("layers") or []:
        if layer is None:
            continue
        alpha = layer[:, :, 3] if layer.ndim == 3 and layer.shape[2] == 4 else layer[:, :, 0]
        mask[alpha > 8] = 255
    return mask if mask.any() else None


def _engine_kwargs(engine: str) -> dict:
    path = MODEL_PATHS.get(engine)
    if path is not None and not path.exists():
        raise gr.Error(f"Model {engine} belum ada. "
                       f"Jalankan: uv run wmr download-model --model {engine}")
    return {}


def process_image(editor_value, engine: str, grow: int):
    mask = mask_from_editor(editor_value)
    if mask is None:
        raise gr.Error("Belum ada area yang dicoret. Pakai brush untuk menandai watermark.")

    src = OUT_DIR / "input.png"
    dst = OUT_DIR / "clean.png"
    write_image(src, editor_value["background"][:, :, :3])
    info = remove_from_image(src, dst, mask, engine=engine, grow=int(grow),
                             **_engine_kwargs(engine))

    region = bbox(dilate(mask, int(grow)))
    note = f"Engine {info['engine']} - {info['masked_pixels']:,} px dibersihkan"
    if region:
        note += f" - area {region.w}x{region.h} @ ({region.x},{region.y})"
    return str(dst), str(dst), note


def load_video_frame(path: str | None):
    if not path:
        return None, "Upload video dulu."
    info = ff.probe(path)
    frame = ff.grab_frame(path, at_seconds=min(1.0, max(0.0, info.duration / 3)))
    detail = (f"{info.width}x{info.height} - {info.fps:.2f} fps - "
              f"{info.n_frames} frame - {info.duration:.1f}s - "
              f"audio: {'ada' if info.has_audio else 'tidak ada'}")
    return frame, detail


def process_video(video_path, editor_value, engine: str, grow: int, crf: int,
                  moving: bool, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload video dulu.")
    mask = mask_from_editor(editor_value)
    if mask is None:
        raise gr.Error("Belum ada area yang dicoret di frame preview.")

    info = ff.probe(video_path)
    reference = editor_value["background"]
    if mask.shape != (info.height, info.width):
        mask = cv2.resize(mask, (info.width, info.height), interpolation=cv2.INTER_NEAREST)
        reference = cv2.resize(reference[:, :, :3], (info.width, info.height),
                               interpolation=cv2.INTER_AREA)

    if moving and engine == "delogo":
        raise gr.Error("Engine delogo hanya bisa satu kotak tetap. "
                       "Pilih migan (atau telea/ns/lama) untuk watermark bergerak.")

    dst = OUT_DIR / f"clean-{Path(video_path).stem}.mp4"
    progress(0, desc="Menyiapkan...")

    track = None
    extra = ""
    if moving:
        seed = bbox(mask)
        if seed is None:
            raise gr.Error("Mask kosong - tidak ada yang bisa dilacak.")

        def on_track(done: int, total: int) -> None:
            progress(0.5 * done / total if total else 0,
                     desc=f"Melacak watermark {done}/{total or '?'}")

        try:
            track = track_watermark(video_path, seed, reference_frame=reference[:, :, :3],
                                    progress=on_track)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        quality = track_quality(track)
        extra = (f" - dilacak, x {quality['x_range']}, y {quality['y_range']}"
                 if quality["moved"] else " - dilacak (watermark ternyata diam)")

    def on_frame(done: int, total: int) -> None:
        base = 0.5 if moving else 0.0
        span = 0.5 if moving else 1.0
        progress(base + span * (done / total if total else 0),
                 desc=f"Frame {done}/{total or '?'}")

    result = remove_from_video(video_path, dst, mask, engine=engine, grow=int(grow),
                               crf=int(crf), track=track,
                               progress=None if (engine == "delogo" and not moving) else on_frame,
                               **_engine_kwargs(engine))
    note = (f"Engine {result['engine']} - {result.get('frames', 0)} frame - "
            f"audio {'ikut' if result['audio'] else 'tidak ada'}{extra}")
    return str(dst), str(dst), note


CSS = """
.wmr-note { font-size: 0.9rem; opacity: 0.85; }
footer { display: none !important; }
"""

with gr.Blocks(title="WMR Studio") as demo:
    gr.Markdown(
        "# WMR Studio\n"
        "Hapus watermark / logo dari **gambar** dan **video**. "
        "Coret area watermark dengan brush, pilih engine, lalu proses. "
        "Semua diproses lokal di PC ini."
    )

    with gr.Tab("Gambar"):
        with gr.Row():
            with gr.Column(scale=3):
                img_editor = gr.ImageEditor(
                    label="Upload gambar, lalu coret watermark-nya",
                    type="numpy", height=460,
                    brush=gr.Brush(colors=["#ff2d55"], color_mode="fixed", default_size=28),
                    layers=False, sources=["upload", "clipboard"],
                )
            with gr.Column(scale=2):
                img_engine = gr.Dropdown(IMAGE_ENGINES, value=IMAGE_ENGINES[0],
                                         label="Engine")
                img_grow = gr.Slider(0, 24, value=3, step=1, label="Perbesar mask (px)")
                img_run = gr.Button("Hapus watermark", variant="primary")
                img_note = gr.Markdown("", elem_classes="wmr-note")
        with gr.Row():
            img_out = gr.Image(label="Hasil", height=420)
            img_file = gr.File(label="Download")

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column(scale=2):
                vid_in = gr.Video(label="Upload video", height=300)
                vid_info = gr.Markdown("", elem_classes="wmr-note")
                vid_engine = gr.Dropdown(VIDEO_ENGINES, value=VIDEO_ENGINES[0],
                                         label="Engine")
                vid_grow = gr.Slider(0, 24, value=3, step=1, label="Perbesar mask (px)")
                vid_crf = gr.Slider(12, 28, value=18, step=1,
                                    label="Kualitas encode (CRF, kecil = bagus)")
                vid_moving = gr.Checkbox(
                    value=False, label="Watermark bergerak (lacak otomatis)",
                    info="Centang kalau logonya pindah posisi / loncat sudut. "
                         "Butuh satu pass tambahan untuk melacak.")
                vid_run = gr.Button("Hapus watermark", variant="primary")
                vid_note = gr.Markdown("", elem_classes="wmr-note")
            with gr.Column(scale=3):
                vid_editor = gr.ImageEditor(
                    label="Coret watermark di frame ini (berlaku untuk seluruh video)",
                    type="numpy", height=420,
                    brush=gr.Brush(colors=["#ff2d55"], color_mode="fixed", default_size=28),
                    layers=False, sources=[],
                )
        with gr.Row():
            vid_out = gr.Video(label="Hasil", height=360)
            vid_file = gr.File(label="Download")

    gr.Markdown(
        "**Engine** — `migan`: AI inpainting di GPU (DirectML), cepat + kualitas bagus — "
        "pilihan default. `delogo`: ffmpeg, real-time, untuk logo kecil semi-transparan. "
        "`telea` / `ns`: inpainting klasik OpenCV. "
        "`lama`: kualitas tertinggi untuk area besar, tapi CPU-only jadi lambat."
    )

    img_run.click(process_image, [img_editor, img_engine, img_grow],
                  [img_out, img_file, img_note])
    vid_in.change(load_video_frame, vid_in, [vid_editor, vid_info])
    vid_run.click(process_video,
                  [vid_in, vid_editor, vid_engine, vid_grow, vid_crf, vid_moving],
                  [vid_out, vid_file, vid_note])

if __name__ == "__main__":
    demo.launch(inbrowser=False, server_port=7861,
                css=CSS, theme=gr.themes.Soft())

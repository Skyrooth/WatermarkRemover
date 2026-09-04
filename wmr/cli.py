"""Command line interface: wmr image / video / probe / download-model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__, ffmpeg as ff
from .image_ops import read_image, remove_from_image
from .mask import bbox, load_mask, mask_from_boxes, parse_box
from .tracking import track_quality, track_watermark
from .video_ops import remove_from_video

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _target_shape(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        rgb, _ = read_image(path)
        return rgb.shape[:2]
    info = ff.probe(path)
    return info.height, info.width


def _build_mask(args, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if args.mask:
        return load_mask(args.mask, shape)
    if args.box:
        boxes = [parse_box(spec, width, height) for spec in args.box]
        return mask_from_boxes(shape, boxes)
    raise SystemExit("Give a region with --box x,y,w,h (repeatable) or --mask file.png")


def _default_output(src: Path, suffix: str = "-clean") -> Path:
    return src.with_name(f"{src.stem}{suffix}{src.suffix}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--box", action="append", metavar="X,Y,W,H",
                        help="Region to remove; percentages allowed (70%%,85%%,25%%,10%%)")
    parser.add_argument("--mask", type=Path, help="Mask image; white = remove")
    parser.add_argument("--grow", type=int, default=3,
                        help="Dilate the mask by N px before inpainting (default 3)")
    parser.add_argument("--model", type=Path, help="Override the .onnx model path")
    parser.add_argument("--max-side", type=int, default=512,
                        help="Resolution fed to the ONNX model (default 512)")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wmr", description="Remove watermarks and logos from images and video.")
    parser.add_argument("-V", "--version", action="version", version=f"wmr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("image", help="Remove a watermark from an image")
    _add_common(image)
    # Stills default to lama: it scores about twice as well as anything else and a
    # one-off 2.5s is cheap. Video defaults to migan, where per-frame speed decides.
    image.add_argument("--engine", default="lama",
                       choices=("telea", "ns", "migan", "lama"))

    video = sub.add_parser("video", help="Remove a watermark from a video")
    _add_common(video)
    video.add_argument("--engine", default="migan",
                       choices=("delogo", "telea", "ns", "migan", "lama"))
    video.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better")
    video.add_argument("--preset", default="medium", help="x264 speed preset")
    video.add_argument("--track", action="store_true",
                       help="Follow a moving watermark instead of using one fixed box")
    video.add_argument("--at", type=float, default=0.0, metavar="SECONDS",
                       help="Timestamp the --box/--mask describes (default 0)")
    video.add_argument("--track-search", type=int, default=120, metavar="PX",
                       help="Local search radius per frame (default 120)")
    video.add_argument("--track-threshold", type=float, default=0.45,
                       help="Match score below which a frame is interpolated (default 0.45)")
    video.add_argument("--track-smooth", type=int, default=3, metavar="FRAMES",
                       help="Median window over positions, 0 disables (default 3)")

    probe = sub.add_parser("probe", help="Show media info")
    probe.add_argument("input", type=Path)

    download = sub.add_parser("download-model", help="Fetch an ONNX inpainting model")
    download.add_argument("--model", default="migan", choices=("migan", "lama"))
    return parser


def _engine_kwargs(args) -> dict:
    kwargs = {"max_side": args.max_side}
    if args.model:
        kwargs["model_path"] = args.model
    return kwargs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download-model":
        from .download import download_model
        print(download_model(args.model))
        return 0

    if args.command == "probe":
        path = Path(args.input)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            rgb, alpha = read_image(path)
            print(json.dumps({"type": "image", "width": rgb.shape[1],
                              "height": rgb.shape[0], "alpha": alpha is not None}, indent=2))
        else:
            print(json.dumps(ff.probe(path).__dict__, indent=2))
        return 0

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")
    dst = Path(args.output) if args.output else _default_output(src)
    mask = _build_mask(args, _target_shape(src))

    if args.command == "image":
        result = remove_from_image(src, dst, mask, engine=args.engine,
                                   grow=args.grow, **_engine_kwargs(args))
    else:
        def make_progress(label: str):
            def progress(done: int, total: int) -> None:
                total = total or done
                pct = 100.0 * done / total if total else 0.0
                print(f"\r  {label} {done}/{total} ({pct:5.1f}%)", end="", file=sys.stderr)
            return progress

        track = None
        if args.track:
            seed = bbox(mask)
            if seed is None:
                raise SystemExit("Mask is empty - nothing to track")
            reference = ff.grab_frame(src, args.at)
            track = track_watermark(src, seed, search_radius=args.track_search,
                                    threshold=args.track_threshold,
                                    smooth_window=args.track_smooth,
                                    reference_frame=reference,
                                    progress=make_progress("tracking"))
            print(file=sys.stderr)
            quality = track_quality(track)
            print(f"  track: {quality['frames']} frames, "
                  f"x {quality['x_range']}, y {quality['y_range']}, "
                  f"max step {quality['max_step_px']:.1f}px", file=sys.stderr)
            if not quality["moved"]:
                print("  note: the watermark never moved - --track was not needed",
                      file=sys.stderr)

        show_progress = args.engine != "delogo" or track is not None
        result = remove_from_video(src, dst, mask, engine=args.engine, grow=args.grow,
                                   crf=args.crf, preset=args.preset, track=track,
                                   progress=make_progress("frames") if show_progress else None,
                                   **_engine_kwargs(args))
        if show_progress:
            print(file=sys.stderr)

    print(json.dumps(result, indent=2) if args.json else f"Done -> {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke tests. Run with: uv run --with pytest pytest -q"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from wmr import ffmpeg as ff
from wmr.engines import LAMA_PATH, MIGAN_PATH, get_engine
from wmr.image_ops import read_image, remove_from_image, write_image
from wmr.mask import Box, bbox, boxes_from_mask, dilate, mask_from_boxes, parse_box
from wmr.tracking import TemplateTracker, track_quality, track_watermark
from wmr.video_ops import remove_from_video

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SAMPLE_VIDEO = SAMPLES / "wm_static.mp4"
MOVING_VIDEO = SAMPLES / "wm_moving.mp4"
TRUTH_FILE = SAMPLES / "truth.json"

REGEN = "run: uv run python scripts/make_samples.py"
needs_sample = pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason=REGEN)
needs_moving = pytest.mark.skipif(not (MOVING_VIDEO.exists() and TRUTH_FILE.exists()),
                                  reason=REGEN)

ONNX_ENGINES = [
    pytest.param(name, marks=pytest.mark.skipif(
        not path.exists(), reason=f"run: uv run wmr download-model --model {name}"))
    for name, path in (("migan", MIGAN_PATH), ("lama", LAMA_PATH))
]


def truth() -> dict:
    return json.loads(TRUTH_FILE.read_text(encoding="utf-8"))


LOGO_BOX = "940,590,300,90"


def test_parse_box_absolute_and_percent():
    assert parse_box("10,20,30,40", 1000, 500) == parse_box("1%,4%,3%,8%", 1000, 500)


def test_parse_box_clamps_to_frame():
    box = parse_box("900,400,500,500", 1000, 500)
    assert box.x + box.w <= 1000 and box.y + box.h <= 500


def test_parse_box_rejects_bad_spec():
    with pytest.raises(ValueError):
        parse_box("10,20,30", 100, 100)


def test_mask_roundtrip():
    box = parse_box("100,50,40,20", 400, 300)
    mask = mask_from_boxes((300, 400), [box])
    assert int(mask.sum() // 255) == 40 * 20
    assert bbox(mask) == box
    assert boxes_from_mask(mask) == [box]
    assert dilate(mask, 2).sum() > mask.sum()


def test_opencv_engines_only_touch_the_mask():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), np.uint8)
    mask[40:70, 60:100] = 255

    for name in ("telea", "ns"):
        out = get_engine(name).inpaint(img, mask)
        assert out.shape == img.shape
        assert np.array_equal(out[mask == 0], img[mask == 0]), f"{name} touched clean pixels"
        assert not np.array_equal(out[mask > 0], img[mask > 0]), f"{name} left the mask untouched"


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        get_engine("nope")


@pytest.mark.parametrize("name", ONNX_ENGINES)
def test_onnx_engine_runs_and_leaves_clean_pixels_alone(name):
    """The engine must land on a working provider and repair only the masked area."""
    engine = get_engine(name)
    assert engine.providers, "no execution provider selected"

    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    mask = np.zeros((400, 600), np.uint8)
    mask[180:230, 250:380] = 255

    out = engine.inpaint(img, mask)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out[mask > 0], img[mask > 0])
    # Outside the crop window the frame must be byte-identical to the source.
    assert np.array_equal(out[:100], img[:100])


@pytest.mark.parametrize("name", ONNX_ENGINES)
def test_onnx_empty_mask_is_a_noop(name):
    img = np.full((200, 200, 3), 90, np.uint8)
    out = get_engine(name).inpaint(img, np.zeros((200, 200), np.uint8))
    assert np.array_equal(out, img)


@pytest.mark.parametrize("name", ONNX_ENGINES)
def test_onnx_engine_actually_removes_a_solid_block(name):
    """A bright block on a smooth background must come back close to its surroundings."""
    base = np.tile(np.linspace(40, 200, 512, dtype=np.uint8)[:, None], (1, 512))
    img = np.dstack([base, base, base]).copy()
    truth = img.copy()
    img[200:260, 180:340] = 255
    mask = np.zeros((512, 512), np.uint8)
    mask[200:260, 180:340] = 255

    out = get_engine(name).inpaint(img, mask)
    before = float(np.abs(img[200:260, 180:340].astype(int)
                          - truth[200:260, 180:340].astype(int)).mean())
    after = float(np.abs(out[200:260, 180:340].astype(int)
                         - truth[200:260, 180:340].astype(int)).mean())
    assert after < before / 3, f"{name}: block not removed (err {before:.1f} -> {after:.1f})"


def test_image_alpha_is_preserved(tmp_path):
    rgb = np.full((60, 80, 3), 128, np.uint8)
    alpha = np.full((60, 80), 200, np.uint8)
    src = tmp_path / "in.png"
    write_image(src, rgb, alpha)

    mask = np.zeros((60, 80), np.uint8)
    mask[20:40, 30:50] = 255
    dst = tmp_path / "out.png"
    remove_from_image(src, dst, mask, engine="telea")

    _, out_alpha = read_image(dst)
    assert out_alpha is not None and int(out_alpha.mean()) == 200


def test_image_rejects_mismatched_mask(tmp_path):
    src = tmp_path / "in.png"
    write_image(src, np.zeros((40, 40, 3), np.uint8))
    with pytest.raises(ValueError):
        remove_from_image(src, tmp_path / "out.png", np.zeros((10, 10), np.uint8))


@needs_sample
def test_probe_reads_stream_metadata():
    info = ff.probe(SAMPLE_VIDEO)
    assert (info.width, info.height) == (1280, 720)
    assert info.n_frames > 0 and info.has_audio


@needs_sample
def test_grab_frame_shape():
    frame = ff.grab_frame(SAMPLE_VIDEO, at_seconds=1.0)
    assert frame.shape == (720, 1280, 3)


@needs_sample
@pytest.mark.parametrize("engine", ["delogo", "telea"])
def test_video_removal_keeps_audio_and_frame_count(tmp_path, engine):
    info = ff.probe(SAMPLE_VIDEO)
    mask = mask_from_boxes((info.height, info.width),
                           [parse_box(LOGO_BOX, info.width, info.height)])
    dst = tmp_path / f"out-{engine}.mp4"
    result = remove_from_video(SAMPLE_VIDEO, dst, mask, engine=engine)

    assert dst.exists() and dst.stat().st_size > 0
    assert result["audio"] is True
    out_info = ff.probe(dst)
    assert (out_info.width, out_info.height) == (info.width, info.height)
    assert abs(out_info.n_frames - info.n_frames) <= 2

    audio = subprocess.run(
        [ff.FFPROBE, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(dst)],
        capture_output=True, text=True).stdout.strip()
    assert audio, "audio stream missing from output"


@needs_sample
def test_watermark_region_actually_changes(tmp_path):
    """The logo area must differ from the source; everything else must not."""
    import cv2

    info = ff.probe(SAMPLE_VIDEO)
    box = parse_box(LOGO_BOX, info.width, info.height)
    mask = mask_from_boxes((info.height, info.width), [box])
    dst = tmp_path / "clean.mp4"
    remove_from_video(SAMPLE_VIDEO, dst, mask, engine="telea", grow=0)

    before = ff.grab_frame(SAMPLE_VIDEO, 2.0)
    after = ff.grab_frame(dst, 2.0)
    patch_diff = cv2.absdiff(before, after)[box.y:box.y + box.h, box.x:box.x + box.w]
    assert patch_diff.mean() > 5, "watermark region unchanged"


# --- moving watermark -------------------------------------------------------


@needs_moving
def test_tracker_follows_the_watermark_exactly():
    """Ground truth comes from make_samples.py, so this is an absolute pixel check."""
    meta = truth()
    want = [Box(*b) for b in meta["moving_boxes"]]
    first = ff.grab_frame(MOVING_VIDEO, 0.0)

    track = track_watermark(MOVING_VIDEO, want[0], reference_frame=first)
    assert len(track) == len(want)

    errors = np.array([float(np.hypot(g.x - w.x, g.y - w.y))
                       for g, w in zip(track, want)])
    assert errors.mean() <= 1.5, f"mean tracking error {errors.mean():.2f}px"
    assert (errors <= 15).all(), f"worst tracking error {errors.max():.2f}px"


@needs_moving
def test_tracker_reports_movement():
    meta = truth()
    want = [Box(*b) for b in meta["moving_boxes"]]
    track = track_watermark(MOVING_VIDEO, want[0],
                            reference_frame=ff.grab_frame(MOVING_VIDEO, 0.0))
    quality = track_quality(track)
    assert quality["moved"] is True
    assert quality["frames"] == len(want)


@needs_sample
def test_tracker_reports_a_static_watermark_as_still():
    meta = truth()
    box = Box(*meta["static_box"])
    track = track_watermark(SAMPLE_VIDEO, box,
                            reference_frame=ff.grab_frame(SAMPLE_VIDEO, 0.0))
    quality = track_quality(track)
    assert quality["moved"] is False, "a fixed watermark must not look like it moved"


@needs_moving
def test_smoothing_keeps_direction_changes():
    """A median filter that replaces every point would flatten a bounce vertex."""
    meta = truth()
    want = [Box(*b) for b in meta["moving_boxes"]]
    track = track_watermark(MOVING_VIDEO, want[0],
                            reference_frame=ff.grab_frame(MOVING_VIDEO, 0.0))
    turns = [i for i in range(1, len(want) - 1)
             if (want[i].x - want[i - 1].x) * (want[i + 1].x - want[i].x) < 0]
    assert turns, "sample has no direction change to check"
    for i in turns:
        assert abs(track[i].x - want[i].x) <= 2, f"vertex at frame {i} was flattened"


@needs_moving
def test_tracked_removal_covers_every_frame():
    meta = truth()
    want = [Box(*b) for b in meta["moving_boxes"]]
    track = track_watermark(MOVING_VIDEO, want[0],
                            reference_frame=ff.grab_frame(MOVING_VIDEO, 0.0))
    seed = mask_from_boxes((meta["height"], meta["width"]), [want[0]])

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "clean.mp4"
        result = remove_from_video(MOVING_VIDEO, dst, seed, engine="telea",
                                   grow=4, track=track)
        assert result["tracked"] is True
        assert result["frames"] == len(want)
        assert dst.exists() and dst.stat().st_size > 0

        # The watermark must be gone wherever it was, not just at its first position.
        for index in (10, 90, 170):
            box = want[index]
            before = ff.grab_frame(MOVING_VIDEO, index / meta["fps"])
            after = ff.grab_frame(dst, index / meta["fps"])
            patch = np.abs(after[box.y:box.y + box.h, box.x:box.x + box.w].astype(int)
                           - before[box.y:box.y + box.h, box.x:box.x + box.w].astype(int))
            assert patch.mean() > 5, f"frame {index}: watermark region untouched"


@needs_moving
def test_delogo_refuses_to_track():
    meta = truth()
    seed = mask_from_boxes((meta["height"], meta["width"]), [Box(*meta["moving_boxes"][0])])
    with pytest.raises(ValueError, match="cannot follow a moving watermark"):
        remove_from_video(MOVING_VIDEO, Path("unused.mp4"), seed,
                          engine="delogo", track=[Box(*meta["moving_boxes"][0])])


@needs_moving
def test_tracker_finds_a_hop_across_the_frame():
    """A corner-to-corner jump must be caught by the full-frame fallback search."""
    meta = truth()
    frame = ff.grab_frame(MOVING_VIDEO, 0.0)
    box = Box(*meta["moving_boxes"][0])
    tracker = TemplateTracker(frame, box)

    # Pretend the last hit was on the opposite side of the frame.
    far = Box(10, 10, box.w, box.h)
    found = tracker.locate(frame, far)
    assert found.confident
    assert abs(found.box.x - box.x) <= 3 and abs(found.box.y - box.y) <= 3


@pytest.mark.parametrize("name", ONNX_ENGINES)
def test_onnx_probe_does_not_leak_runtime_errors(name, capfd):
    """A provider that cannot run the model must fall back quietly, not print a trace."""
    engine = get_engine(name)
    captured = capfd.readouterr()
    assert "Non-zero status code" not in captured.err, (
        f"{name}: onnxruntime error leaked to stderr during the provider probe")
    # Falling back is fine; silently ending up on no provider is not.
    assert engine.providers
    if engine.fallback_from:
        assert engine.providers[0] == "CPUExecutionProvider"


def test_cli_defaults_match_the_measured_best():
    """Stills default to the most accurate engine, video to the one fast enough."""
    from wmr.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["image", "x.png", "--box", "0,0,4,4"]).engine == "lama"
    assert parser.parse_args(["video", "x.mp4", "--box", "0,0,4,4"]).engine == "migan"
    assert parser.parse_args(["video", "x.mp4", "--box", "0,0,4,4"]).track is False

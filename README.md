# WMR — Watermark Remover (images + video)

> 🇮🇩 [Versi Bahasa Indonesia](README_id.md)

Remove watermarks and logos from **images** and **video**. You mark the watermark
(brush, or box coordinates) and the tool *inpaints* that area. Everything runs on your
own machine — nothing is uploaded anywhere.

> **Browser version: [skyrooth.github.io/WatermarkRemover](https://skyrooth.github.io/WatermarkRemover/)**
> Nothing to install. Images go through MI-GAN via `onnxruntime-web`; video runs
> ffmpeg's `delogo` filter as WebAssembly. Both work entirely on your device.

## Why this is not a fork of gemini-watermark-remover

[`GargantuaX/gemini-watermark-remover`](https://github.com/GargantuaX/gemini-watermark-remover)
uses *reverse alpha blending* against **embedded per-pixel alpha maps of the Gemini
logo**. It inverts `output = original×(1−α) + logo×α`, which is exact **only because α
and the logo colour are known in advance**. For an arbitrary logo — TikTok, CapCut,
a stock-photo mark — α is unknown, so none of that engine generalises. The right
approach for arbitrary watermarks is **mask + inpainting**, which is what this does.

## Install

```bash
uv sync
```

Needs **ffmpeg + ffprobe** on PATH.

## GUI

```bash
uv run python app.py
```

Then open http://localhost:7861

- **Image tab** — load an image, paint over the watermark, pick an engine, remove.
- **Video tab** — load a video, a preview frame appears, paint the watermark on it.
  Tick **"Watermark bergerak"** if the watermark moves. The original audio is copied
  across, never re-encoded.

## CLI

```bash
# image (defaults to the lama engine)
uv run wmr image photo.png --box 940,590,300,90 -o clean.png

# coordinates may be percentages
uv run wmr image photo.png --box "72%,82%,24%,13%" --engine ns

# a mask image instead of a box; white = remove
uv run wmr image photo.png --mask mask.png

# video (defaults to migan on the GPU)
uv run wmr video clip.mp4 --box 940,590,300,90 -o clean.mp4

# fastest path: ffmpeg delogo
uv run wmr video clip.mp4 --box 940,590,300,90 --engine delogo

# a MOVING watermark, tracked per frame
uv run wmr video clip.mp4 --box 940,590,300,90 --track

# when the box was taken at 3s rather than the start
uv run wmr video clip.mp4 --box 940,590,300,90 --track --at 3

# media info
uv run wmr probe clip.mp4
```

| Flag | Meaning |
|---|---|
| `--box X,Y,W,H` | Watermark region. Repeatable for several watermarks. Percentages allowed. |
| `--mask file.png` | Mask image instead of a box; white means remove. |
| `--engine` | images: `lama` (default) · video: `migan` (default) · `delogo` (video only) · `telea` · `ns` |
| `--grow N` | Dilate the mask by N px before inpainting (default 3). |
| `--crf N` | Video encode quality, lower is better (default 18). |
| `--track` | Follow a moving watermark (video). |
| `--at S` | Timestamp the `--box`/`--mask` describes (default 0). |
| `--json` | Print the result as JSON. |

## Engines

Measured on a GTX 1660 SUPER at 720p (`scripts/bench_engines.py`):

| Engine | Device | Speed | Best for |
|---|---|---|---|
| `lama` | CPU only | 2.57 s/frame | **Image default.** Highest quality, consistently. |
| `migan` | **GPU (DirectML)** | **0.091 s/frame** | **Video default.** AI inpainting fast enough for clips. |
| `delogo` | CPU (ffmpeg) | real time | Small semi-transparent logos in a fixed position. No `--track`. |
| `telea` / `ns` | CPU | 0.055 s/frame | Fast, and surprisingly strong on smooth backgrounds. |

A 6-second 720p clip: `migan` **26 s**, `lama` **7.6 min**.

### Measured residue

`scripts/eval_removal.py` compares the output against a clean reference clip and
reports the leftover error inside the watermark region, as a share of the original:

| Engine | Smooth background | Textured background |
|---|---|---|
| `lama` | **21.4%** | **24.9%** |
| `telea` | 40.1% | 34.8% |
| `ns` | 39.8% | 35.0% |
| `migan` | 48.0% | 42.9% |

**Read this carefully.** `lama` winning everywhere is solid. But `telea`/`ns` beating
`migan` here is an artefact of the test corpus: both synthetic backgrounds are locally
smooth, which is the ideal terrain for diffusion inpainting. On real footage with
structure, diffusion smears and AI reconstructs — visible in the Veo test below, where
`delogo` leaves horizontal streaks on a detailed frame and `migan` does not. These
numbers settle "lama is the most accurate" and "a small `--grow` beats a large one";
they do not settle `migan` vs `telea` for real video.

## Gemini / Veo watermarks

The Veo mark is a small four-point sparkle in the bottom-right corner. On a 1280×720
clip it measures **48×48 px, inset 96 px from both the right and bottom edges**.

`scripts/find_watermark.py` locates a constant watermark from the video itself — a mark
present in every frame leaves a brightness floor the moving content cannot go below:

```bash
uv run python scripts/find_watermark.py clip.mp4
```

Then remove it. `delogo` is instant and fine over smooth backgrounds; `migan` is
noticeably cleaner when there is texture behind the mark:

```bash
uv run wmr video clip.mp4 --box 1136,576,48,48 --engine migan
```

Measured on a real 10-second Veo clip: `delogo` 2.4 s, `migan` 29 s, both preserving
audio, duration and resolution.

## Models

```bash
uv add onnxruntime-directml
uv run wmr download-model --model migan   # 30 MB
uv run wmr download-model --model lama    # 208 MB
```

**Why `migan` runs on the GPU and `lama` does not.** LaMa uses fast Fourier
convolutions, and DirectML has no kernel for its FFT ops — the graph loads and then
dies at run time. MI-GAN has no FFT and runs fine. Every ONNX engine **proves its
provider with one forward pass** at construction and quietly falls back to the CPU if
that fails (see `OnnxInpaintEngine.fallback_from`), so a long video never dies halfway
through. Running LaMa on a GPU needs the CUDA path: `onnxruntime-gpu` + CUDA 12 +
cuDNN 9.

Two more differences handled automatically: the mask polarity is **inverted** between
them (LaMa 255 = hole, MI-GAN 255 = keep), and the LaMa export is fixed at 512×512
while MI-GAN is dynamic.

## Moving watermarks (`--track`)

When the logo slides around or hops between corners, one static mask is not enough.
`--track` runs two passes:

1. **Track** — template matching on *gradient magnitude* rather than raw colour: the
   logo's edges stay put even as the background under a semi-transparent overlay
   changes. Each frame searches near the previous hit first (cheap); when the score
   drops it sweeps the whole frame, which is what catches a corner hop.
2. **Remove** — the mask is placed at the tracked position on every frame.

Frames that never matched confidently are **interpolated** between the frames that did,
rather than guessed.

**Measured accuracy.** `samples/truth.json` records the true box of every frame, since
the corpus is generated rather than found:

```
uv run python scripts/eval_tracking.py
→ mean 0.00px   median 0.00px   p95 0.00px   max 0.00px   (180 frames, 2 clips)
```

One trap already fixed: a median filter meant to damp jitter also **flattens direction
changes**. For a bouncing logo `median([8, 1, 8])` is `8`, so the bounce vertex
disappears and the mask lags. The median now only *rejects outliers* — deviations
beyond the clip's own motion scale — instead of replacing every position. That is what
took the error from 0.12 px to 0.00 px.

`--engine delogo` cannot be combined with `--track`: that ffmpeg filter draws one fixed
box for the whole clip. The tool refuses with a clear message.

## Browser version (`docs/`)

Images use MI-GAN through `onnxruntime-web`; video runs ffmpeg's `delogo` as
WebAssembly with the audio stream-copied. Both runtimes are **served from the site
itself**, not a CDN — which also makes "nothing leaves your device" literally true.

Notes for anyone reusing this:

- The `@ffmpeg/ffmpeg` ESM worker is a module with relative imports. Handed over as a
  cross-origin blob URL it fails to spawn, and `ffmpeg.load()` then **hangs with no
  error at all**. Same origin avoids it.
- `coreURL` and `wasmURL` must be **absolute**: the worker resolves them against its
  own location, not the page.
- Load plain `ort.min.js`, not `ort.webgpu.min.js`, unless
  `navigator.gpu.requestAdapter()` actually returns an adapter — the WebGPU bundle's
  wasm binary is ~4× slower on the CPU path. `navigator.gpu` merely existing is not
  enough.
- Threaded wasm needs COOP/COEP headers that GitHub Pages cannot send, so both
  runtimes are single-threaded.

## Layout

```
wmr/
  ffmpeg.py     probing, raw frame pipes, audio-preserving mux
  mask.py       boxes (absolute/percent), mask images, dilation, components
  engines.py    OpenCV (telea/ns) + OnnxInpaintEngine → MI-GAN & LaMa
  tracking.py   template tracker for moving watermarks
  image_ops.py  image read/write (alpha preserved) + image pipeline
  video_ops.py  delogo, per-frame and tracked backends
  download.py   model downloads
  cli.py        the wmr command
app.py          Gradio GUI
docs/           the browser version (GitHub Pages)
scripts/        make_samples, find_watermark, eval_tracking, eval_removal, bench_engines
tests/          29 pytest tests
```

### Implementation detail worth knowing

`OnnxInpaintEngine` does not send whole frames to the model. It takes a **crop around
the mask** (enough context, not a whole 4K frame), widens it to the model's aspect
ratio, resizes, inpaints, resizes back, and **feathers the seam**.

One trap already fixed: resizing the crop smears the watermark 1–2 px past its own
edge. With a NEAREST-resized mask that halo sits **outside** the hole, the model reads
it as valid context and pulls it back into the fill. The effect is large — test error
45.9 vs 1.3. The mask is therefore resized with INTER_LINEAR (any partial coverage
counts as hole) and then dilated 2 px.

## Tests and evaluation

```bash
uv run --with pytest pytest -q                    # 29 tests
uv run python scripts/make_samples.py             # rebuild the test corpus
uv run python scripts/eval_tracking.py            # tracking accuracy, in pixels
uv run python scripts/eval_removal.py             # residue vs a clean reference
uv run python scripts/bench_engines.py            # per-engine speed
```

The corpus is generated in numpy rather than with ffmpeg's synthetic sources:
`gradients` is random per run and `testsrc2` is worst-case for inpainting. More
importantly, every watermarked clip has a **pixel-identical clean twin**, so "leftover
watermark" is a number rather than an impression. Two backgrounds exist on purpose —
a smooth one flatters diffusion and LaMa's Fourier convolutions, and benchmarking only
on that would pick the wrong default.

## Notes

- Only the **visible** watermark is removed. Invisible ones such as Google's SynthID
  are untouched.
- Mark watermarks **tightly**. Raising `--grow` from 2 to 8 makes results worse across
  every engine — a bigger hole means more for the model to invent.
- Use this on content you own or have permission to edit.

## License

MIT — see [LICENSE](LICENSE). Third-party models keep their own licenses.

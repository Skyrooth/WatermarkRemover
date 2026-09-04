/* WMR web — watermark removal in the browser, no upload.
 *
 * Images  : MI-GAN inpainting through onnxruntime-web.
 * Video   : ffmpeg's own delogo filter compiled to WebAssembly.
 *
 * The image path mirrors wmr/engines.py — crop around the mask, match the model
 * aspect, resize, inpaint, resize back, feather the seam — including the halo fix:
 * resampling smears the watermark a pixel or two past its own edge, and any of that
 * left outside the hole gets read as valid context and pulled back into the fill.
 */

const ORT_DIR = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ORT_CPU = ORT_DIR + "ort.min.js";
const ORT_GPU = ORT_DIR + "ort.webgpu.min.js";
const MODEL_URL = "models/migan.onnx";

// ffmpeg is served from this site, not a CDN. Its worker is a module with relative
// imports, so handing it over as a cross-origin blob URL makes the worker fail to
// spawn at all — and ffmpeg.load() then hangs with no error. Same-origin avoids the
// whole class of problem, and keeps the "nothing leaves your device" claim literal.
const FFMPEG_DIR = "ffmpeg/";
const FFMPEG_LOAD_TIMEOUT_MS = 120000;

const MODEL_SIZE = 512;
const MARGIN_RATIO = 0.75;
const MIN_MARGIN = 64;
const HALO_DILATE = 2;
const FEATHER_PX = 1.6;

/* ---------- language ------------------------------------------------------ */

const LANG_KEY = "wmr-lang";

function currentLang() {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === "en" || saved === "id") return saved;
  } catch {
    /* private mode: fall through to the browser's own preference */
  }
  return navigator.language?.toLowerCase().startsWith("id") ? "id" : "en";
}

/** Translate a key. English is the source language, so it falls back to itself. */
function t(key) {
  const lang = state?.lang ?? "en";
  return (I18N[lang] && I18N[lang][key]) || EN_FALLBACK[key] || key;
}

const el = (id) => document.getElementById(id);
const base = el("base");
const overlay = el("overlay");
const baseCtx = base.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlay.getContext("2d", { willReadFrequently: true });

const state = {
  lang: "en",
  mode: "image",
  strokes: [],
  current: null,
  drawing: false,
  session: null,
  backend: null,
  loadingSession: null,
  ffmpeg: null,
  loadingFfmpeg: null,
  videoFile: null,
  resultUrl: null,
  resultBlob: null,
  resultName: "wmr-clean.png",
};

/* ---------- status and errors -------------------------------------------- */

function setStatus(text, fraction) {
  el("status").hidden = false;
  el("statusText").textContent = text;
  el("barFill").style.width = `${Math.round(Math.min(1, Math.max(0, fraction ?? 0)) * 100)}%`;
}

function clearStatus() {
  el("status").hidden = true;
}

function showError(message) {
  let box = el("errorBox");
  if (!box) {
    box = document.createElement("div");
    box.id = "errorBox";
    box.className = "error";
    el("engineNote").before(box);
  }
  box.textContent = message;
  box.hidden = false;
}

function hideError() {
  const box = el("errorBox");
  if (box) box.hidden = true;
}

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

/** Yield to the browser so the status can repaint.
 *
 * A background tab never paints, so `requestAnimationFrame` alone would never fire
 * and the whole pipeline would stall the moment someone switches tabs. The timer is
 * the guarantee that this always resolves.
 */
const nextFrame = () => new Promise((resolve) => {
  let settled = false;
  const finish = () => {
    if (settled) return;
    settled = true;
    resolve();
  };
  requestAnimationFrame(finish);
  setTimeout(finish, 250);
});

/* ---------- mask maths (ports of wmr/mask.py and wmr/engines.py) ---------- */

/** Separable max filter: dilating a binary mask by `radius` pixels. */
function dilate(mask, w, h, radius) {
  if (radius <= 0) return mask;

  const sweep = (src, width, height) => {
    const out = new Uint8Array(src.length);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      for (let x = 0; x < width; x++) {
        const lo = Math.max(0, x - radius);
        const hi = Math.min(width - 1, x + radius);
        let hit = 0;
        for (let i = lo; i <= hi && !hit; i++) hit = src[row + i];
        out[row + x] = hit;
      }
    }
    return out;
  };

  const transpose = (src, width, height) => {
    const out = new Uint8Array(src.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) out[x * height + y] = src[y * width + x];
    }
    return out;
  };

  // Horizontal sweep, then the same sweep on the transpose = vertical sweep.
  let work = sweep(mask, w, h);
  work = transpose(work, w, h);
  work = sweep(work, h, w);
  return transpose(work, h, w);
}

function maskBounds(mask, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (mask[row + x]) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

/** Connected components as boxes, so two separate logos stay two separate boxes. */
function maskBoxes(mask, w, h, minArea = 24) {
  const seen = new Uint8Array(mask.length);
  const boxes = [];
  const stack = [];

  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;

    let x0 = start % w, x1 = x0, y0 = (start / w) | 0, y1 = y0, area = 0;
    stack.push(start);
    seen[start] = 1;

    while (stack.length) {
      const index = stack.pop();
      const x = index % w;
      const y = (index / w) | 0;
      area++;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;

      if (x > 0 && mask[index - 1] && !seen[index - 1]) { seen[index - 1] = 1; stack.push(index - 1); }
      if (x < w - 1 && mask[index + 1] && !seen[index + 1]) { seen[index + 1] = 1; stack.push(index + 1); }
      if (y > 0 && mask[index - w] && !seen[index - w]) { seen[index - w] = 1; stack.push(index - w); }
      if (y < h - 1 && mask[index + w] && !seen[index + w]) { seen[index + w] = 1; stack.push(index + w); }
    }

    if (area >= minArea) boxes.push({ x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 });
  }
  return boxes;
}

/** Context window around the mask, then squared off to the model's aspect. */
function cropWindow(bounds, w, h) {
  const bw = bounds.x1 - bounds.x0 + 1;
  const bh = bounds.y1 - bounds.y0 + 1;
  const mx = Math.max(MIN_MARGIN, Math.round(bw * MARGIN_RATIO));
  const my = Math.max(MIN_MARGIN, Math.round(bh * MARGIN_RATIO));

  let x = Math.max(0, bounds.x0 - mx);
  let y = Math.max(0, bounds.y0 - my);
  let cw = Math.min(w, bounds.x1 + 1 + mx) - x;
  let ch = Math.min(h, bounds.y1 + 1 + my) - y;

  if (ch > cw) {
    const target = Math.min(ch, w);
    x = Math.max(0, Math.min(x - ((target - cw) >> 1), w - target));
    cw = target;
  } else {
    const target = Math.min(cw, h);
    y = Math.max(0, Math.min(y - ((target - ch) >> 1), h - target));
    ch = target;
  }
  return { x, y, w: cw, h: ch };
}

/** The painted overlay's alpha channel, as a binary mask grown by `grow` px. */
function currentMask() {
  const w = base.width;
  const h = base.height;
  const painted = overlayCtx.getImageData(0, 0, w, h).data;
  let mask = new Uint8Array(w * h);
  let any = false;
  for (let i = 0, p = 3; i < mask.length; i++, p += 4) {
    if (painted[p] > 8) {
      mask[i] = 1;
      any = true;
    }
  }
  if (!any) throw new Error(t("err.noStrokes"));
  const grow = Number(el("grow").value);
  return grow > 0 ? dilate(mask, w, h, grow) : mask;
}

/* ---------- image path: MI-GAN ------------------------------------------- */

/** Pick the onnxruntime build to load.
 *
 * The WebGPU bundle ships a different wasm binary that is ~4x slower on the CPU path,
 * so loading it "just in case" penalises every visitor without a working GPU. Ask for
 * a real adapter first — `navigator.gpu` existing is not the same as WebGPU working.
 */
async function loadRuntime() {
  let gpu = false;
  try {
    gpu = Boolean(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    gpu = false;
  }

  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = gpu ? ORT_GPU : ORT_CPU;
    script.onload = resolve;
    script.onerror = () => reject(new Error(t("err.ort")));
    document.head.append(script);
  });

  ort.env.wasm.wasmPaths = ORT_DIR;
  ort.env.wasm.numThreads = 1; // threaded wasm needs COOP/COEP, which Pages cannot set
  ort.env.logLevel = "error";
  return gpu;
}

async function loadSession() {
  if (state.session) return state.session;
  if (state.loadingSession) return state.loadingSession;

  state.loadingSession = (async () => {
    const gpu = await loadRuntime();

    setStatus(t("st.downloadModel"), 0);
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`${t("err.model")} (HTTP ${response.status})`);

    const total = Number(response.headers.get("Content-Length")) || 0;
    const chunks = [];
    let received = 0;
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      setStatus(`${t("st.downloading")} ${(received / 1e6).toFixed(1)} MB`,
        total ? (received / total) * 0.9 : 0.5);
    }

    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }

    setStatus(t("st.prepModel"), 0.95);
    for (const backend of gpu ? ["webgpu", "wasm"] : ["wasm"]) {
      try {
        state.session = await ort.InferenceSession.create(bytes.buffer, {
          executionProviders: [backend],
          graphOptimizationLevel: "all",
        });
        state.backend = backend;
        break;
      } catch (error) {
        console.warn(`${backend} unavailable:`, error);
      }
    }
    if (!state.session) throw new Error(t("err.noBackend"));
    return state.session;
  })();

  try {
    return await state.loadingSession;
  } finally {
    state.loadingSession = null;
  }
}

async function inpaintImage() {
  const W = base.width;
  const H = base.height;
  const mask = currentMask();
  const crop = cropWindow(maskBounds(mask, W, H), W, H);

  const session = await loadSession();
  setStatus(t("st.removing"), 0.4);
  await nextFrame();

  const small = makeCanvas(MODEL_SIZE, MODEL_SIZE);
  const smallCtx = small.getContext("2d", { willReadFrequently: true });
  smallCtx.imageSmoothingQuality = "high";
  smallCtx.drawImage(base, crop.x, crop.y, crop.w, crop.h, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const smallPixels = smallCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;

  const maskFull = makeCanvas(W, H);
  const maskFullCtx = maskFull.getContext("2d");
  const maskImage = maskFullCtx.createImageData(W, H);
  for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
    const v = mask[i] ? 255 : 0;
    maskImage.data[p] = maskImage.data[p + 1] = maskImage.data[p + 2] = v;
    maskImage.data[p + 3] = 255;
  }
  maskFullCtx.putImageData(maskImage, 0, 0);

  // The mask takes the same trip, but partial coverage counts as hole and the result
  // is dilated: that keeps the resampling halo out of the model's context.
  const maskSmall = makeCanvas(MODEL_SIZE, MODEL_SIZE);
  const maskSmallCtx = maskSmall.getContext("2d", { willReadFrequently: true });
  maskSmallCtx.imageSmoothingQuality = "high";
  maskSmallCtx.drawImage(maskFull, crop.x, crop.y, crop.w, crop.h, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const maskSmallPixels = maskSmallCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;

  let hole = new Uint8Array(MODEL_SIZE * MODEL_SIZE);
  for (let i = 0, p = 0; i < hole.length; i++, p += 4) hole[i] = maskSmallPixels[p] > 0 ? 1 : 0;
  hole = dilate(hole, MODEL_SIZE, MODEL_SIZE, HALO_DILATE);

  // MI-GAN's mask polarity is the opposite of LaMa's: 255 means keep this pixel.
  const plane = MODEL_SIZE * MODEL_SIZE;
  const imageTensor = new Uint8Array(3 * plane);
  const maskTensor = new Uint8Array(plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    imageTensor[i] = smallPixels[p];
    imageTensor[plane + i] = smallPixels[p + 1];
    imageTensor[2 * plane + i] = smallPixels[p + 2];
    maskTensor[i] = hole[i] ? 0 : 255;
  }

  const [imageName, maskName] = session.inputNames;
  const output = await session.run({
    [imageName]: new ort.Tensor("uint8", imageTensor, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    [maskName]: new ort.Tensor("uint8", maskTensor, [1, 1, MODEL_SIZE, MODEL_SIZE]),
  });
  const filled = output[session.outputNames[0]].data;

  setStatus(t("st.composing"), 0.85);
  await nextFrame();

  const outSmall = makeCanvas(MODEL_SIZE, MODEL_SIZE);
  const outSmallCtx = outSmall.getContext("2d");
  const outImage = outSmallCtx.createImageData(MODEL_SIZE, MODEL_SIZE);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    outImage.data[p] = filled[i];
    outImage.data[p + 1] = filled[plane + i];
    outImage.data[p + 2] = filled[2 * plane + i];
    outImage.data[p + 3] = 255;
  }
  outSmallCtx.putImageData(outImage, 0, 0);

  const outCrop = makeCanvas(crop.w, crop.h);
  const outCropCtx = outCrop.getContext("2d", { willReadFrequently: true });
  outCropCtx.imageSmoothingQuality = "high";
  outCropCtx.drawImage(outSmall, 0, 0, crop.w, crop.h);
  const repaired = outCropCtx.getImageData(0, 0, crop.w, crop.h);

  const feather = makeCanvas(crop.w, crop.h);
  const featherCtx = feather.getContext("2d", { willReadFrequently: true });
  featherCtx.filter = `blur(${FEATHER_PX}px)`;
  featherCtx.drawImage(maskFull, crop.x, crop.y, crop.w, crop.h, 0, 0, crop.w, crop.h);
  const alpha = featherCtx.getImageData(0, 0, crop.w, crop.h).data;

  const original = baseCtx.getImageData(crop.x, crop.y, crop.w, crop.h);
  for (let p = 0; p < original.data.length; p += 4) {
    const a = alpha[p] / 255;
    for (let c = 0; c < 3; c++) {
      original.data[p + c] = repaired.data[p + c] * a + original.data[p + c] * (1 - a);
    }
  }
  baseCtx.putImageData(original, crop.x, crop.y);

  await new Promise((resolve) => base.toBlob((blob) => {
    state.resultBlob = blob;
    state.resultName = "wmr-clean.png";
    resolve();
  }, "image/png"));
}

/* ---------- video path: ffmpeg delogo ------------------------------------ */

async function loadFfmpeg() {
  if (state.ffmpeg) return state.ffmpeg;
  if (state.loadingFfmpeg) return state.loadingFfmpeg;

  state.loadingFfmpeg = (async () => {
    setStatus(t("st.loadFfmpeg"), 0.02);
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `${FFMPEG_DIR}ffmpeg.js`;
      script.onload = resolve;
      script.onerror = () => reject(new Error(t("err.ffmpegScript")));
      document.head.append(script);
    });

    const ffmpeg = new FFmpegWASM.FFmpeg();
    ffmpeg.on("log", ({ message }) => console.debug("[ffmpeg]", message));

    setStatus(t("st.prepFfmpeg"), 0.1);
    // Absolute URLs: the worker resolves these against its own location, not the page,
    // so a relative path here becomes "Cannot find module".
    const absolute = (name) => new URL(FFMPEG_DIR + name, location.href).href;

    // A bad core/worker pairing hangs instead of throwing, so cap the wait.
    await Promise.race([
      ffmpeg.load({
        coreURL: absolute("ffmpeg-core.js"),
        wasmURL: absolute("ffmpeg-core.wasm"),
      }),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(t("err.ffmpegLoad"))), FFMPEG_LOAD_TIMEOUT_MS)),
    ]);
    state.ffmpeg = ffmpeg;
    return ffmpeg;
  })();

  try {
    return await state.loadingFfmpeg;
  } finally {
    state.loadingFfmpeg = null;
  }
}

/** delogo refuses boxes touching the frame border, so keep 1px of margin. */
function delogoFilter(boxes, width, height) {
  return boxes
    .map((box) => {
      const x = Math.max(1, box.x);
      const y = Math.max(1, box.y);
      const w = Math.max(1, Math.min(box.w, width - x - 1));
      const h = Math.max(1, Math.min(box.h, height - y - 1));
      return `delogo=x=${x}:y=${y}:w=${w}:h=${h}`;
    })
    .join(",");
}

async function processVideo() {
  const mask = currentMask();
  const boxes = maskBoxes(mask, base.width, base.height);
  if (!boxes.length) throw new Error(t("err.tooSmall"));

  const ffmpeg = await loadFfmpeg();

  setStatus(t("st.prepVideo"), 0.3);
  await ffmpeg.writeFile("in.mp4", new Uint8Array(await state.videoFile.arrayBuffer()));

  const onProgress = ({ progress }) =>
    setStatus(`${t("st.processing")} ${Math.round(progress * 100)}%`, 0.35 + progress * 0.6);
  ffmpeg.on("progress", onProgress);

  try {
    const code = await ffmpeg.exec([
      "-i", "in.mp4",
      "-vf", delogoFilter(boxes, base.width, base.height),
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-crf", String(el("crf").value),
      "-pix_fmt", "yuv420p",
      "-c:a", "copy",
      "-movflags", "+faststart",
      "out.mp4",
    ]);
    if (code !== 0) throw new Error(`${t("err.ffmpegRun")} (${code})`);
  } finally {
    ffmpeg.off("progress", onProgress);
  }

  setStatus(t("st.composing"), 0.97);
  const data = await ffmpeg.readFile("out.mp4");
  await ffmpeg.deleteFile("in.mp4").catch(() => {});
  await ffmpeg.deleteFile("out.mp4").catch(() => {});

  state.resultBlob = new Blob([data.buffer], { type: "video/mp4" });
  state.resultName = "wmr-clean.mp4";

  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.resultUrl = URL.createObjectURL(state.resultBlob);
  const player = el("result");
  player.src = state.resultUrl;
  player.hidden = false;
}

/* ---------- corner presets ------------------------------------------------ */

// Watermarks sit in a corner with padding around them, sized relative to the frame.
// These proportions cover the Veo/Gemini sparkle on a 720p clip (48x48, inset 96)
// with room to spare; the mask can always be extended by painting.
const CORNER_SIZE = 0.10;
const CORNER_INSET = 0.11;

function cornerRect(corner) {
  const short = Math.min(base.width, base.height);
  const size = Math.round(short * CORNER_SIZE);
  const inset = Math.round(short * CORNER_INSET);
  const right = corner.endsWith("r");
  const bottom = corner.startsWith("b");
  return {
    x: right ? Math.max(0, base.width - inset - size) : inset,
    y: bottom ? Math.max(0, base.height - inset - size) : inset,
    w: Math.min(size, base.width),
    h: Math.min(size, base.height),
  };
}


/* ---------- painting ----------------------------------------------------- */

function redrawOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.strokeStyle = "rgba(255, 45, 85, 0.55)";
  overlayCtx.fillStyle = "rgba(255, 45, 85, 0.55)";
  overlayCtx.lineCap = "round";
  overlayCtx.lineJoin = "round";

  for (const stroke of state.strokes) {
    if (stroke.rect) {
      overlayCtx.fillRect(stroke.rect.x, stroke.rect.y, stroke.rect.w, stroke.rect.h);
      continue;
    }
    overlayCtx.lineWidth = stroke.size;
    overlayCtx.beginPath();
    stroke.points.forEach(([x, y], i) => (i === 0 ? overlayCtx.moveTo(x, y) : overlayCtx.lineTo(x, y)));
    if (stroke.points.length === 1) {
      overlayCtx.lineTo(stroke.points[0][0] + 0.01, stroke.points[0][1]);
    }
    overlayCtx.stroke();
  }

  const has = state.strokes.length > 0;
  el("undo").disabled = !has;
  el("clear").disabled = !has;
  el("run").disabled = !has;
  // Say why the main button is off, rather than leaving a grey button unexplained.
  el("whyDisabled").hidden = has || el("editor").hidden;
}

function pointerPosition(event) {
  const rect = overlay.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) / rect.width) * overlay.width,
    ((event.clientY - rect.top) / rect.height) * overlay.height,
  ];
}

function brushSize() {
  // The slider is in display pixels; strokes live in source pixels.
  const rect = overlay.getBoundingClientRect();
  return Number(el("brush").value) * (rect.width ? overlay.width / rect.width : 1);
}

overlay.addEventListener("pointerdown", (event) => {
  overlay.setPointerCapture(event.pointerId);
  state.drawing = true;
  state.current = { size: brushSize(), points: [pointerPosition(event)] };
  state.strokes.push(state.current);
  redrawOverlay();
});

overlay.addEventListener("pointermove", (event) => {
  if (!state.drawing) return;
  state.current.points.push(pointerPosition(event));
  redrawOverlay();
});

const endStroke = () => {
  state.drawing = false;
  state.current = null;
};
overlay.addEventListener("pointerup", endStroke);
overlay.addEventListener("pointercancel", endStroke);

/* ---------- loading files ------------------------------------------------ */

function startEditing() {
  el("dropzone").hidden = true;
  el("editor").hidden = false;
  el("reset").hidden = false;
  el("corners").hidden = false;
  redrawOverlay();
  el("download").disabled = true;
  el("result").hidden = true;
  hideError();
  clearStatus();
}

function loadImageFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      state.strokes = [];
      base.width = overlay.width = img.naturalWidth;
      base.height = overlay.height = img.naturalHeight;
      baseCtx.drawImage(img, 0, 0);
      startEditing();
    };
    img.onerror = () => showError(t("err.badImage"));
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function loadVideoFile(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";

  video.onloadeddata = () => {
    // A frame from a third of the way in: more representative than a black opener.
    video.currentTime = Math.min(1, (video.duration || 3) / 3);
  };
  video.onseeked = () => {
    state.strokes = [];
    state.videoFile = file;
    base.width = overlay.width = video.videoWidth;
    base.height = overlay.height = video.videoHeight;
    baseCtx.drawImage(video, 0, 0);
    URL.revokeObjectURL(url);
    startEditing();
    el("hint").textContent =
      `${t("st.maskAllFrames")} — ${video.videoWidth}×${video.videoHeight}, ` +
      `${(video.duration || 0).toFixed(1)} ${t("st.videoInfo")}.`;
  };
  video.onerror = () => {
    URL.revokeObjectURL(url);
    showError(t("err.badVideo"));
  };
  video.src = url;
}

function loadFile(file) {
  if (!file) return;
  hideError();

  const isVideo = file.type.startsWith("video/");
  const isImage = file.type.startsWith("image/");
  if (!isVideo && !isImage) return showError(t("err.pickFile"));

  // Follow the file rather than scolding the user: dropping a video while the Image
  // tab happens to be open should just switch tabs.
  const wanted = isVideo ? "video" : "image";
  if (state.mode !== wanted) setMode(wanted);

  if (isVideo) loadVideoFile(file);
  else loadImageFile(file);
}

/* ---------- mode switching ----------------------------------------------- */

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((tab) =>
    tab.classList.toggle("active", tab.dataset.mode === mode));

  const video = mode === "video";
  // Both types are always accepted so the picker never hides the user's file; the
  // tab then follows whatever they chose.
  el("file").accept = "image/*,video/*";
  el("pick").textContent = t(video ? "pick.video" : "pick.image");
  el("dropIcon").textContent = video ? "🎬" : "🖼️";
  el("crfField").hidden = !video;
  el("hint").textContent = t(video ? "hint.video" : "hint.image");

  state.strokes = [];
  state.videoFile = null;
  el("editor").hidden = true;
  el("dropzone").hidden = false;
  el("reset").hidden = true;
  el("corners").hidden = true;
  el("whyDisabled").hidden = true;
  el("download").disabled = true;
  el("result").hidden = true;
  el("file").value = "";
  hideError();
  clearStatus();
}

document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => setMode(tab.dataset.mode)));

/* ---------- wiring ------------------------------------------------------- */

el("pick").addEventListener("click", () => el("file").click());
el("file").addEventListener("change", (event) => loadFile(event.target.files[0]));

el("demo").addEventListener("click", async () => {
  try {
    const name = state.mode === "video" ? "demo.mp4" : "demo.jpg";
    const response = await fetch(name);
    if (!response.ok) throw new Error();
    const blob = await response.blob();
    loadFile(new File([blob], name, { type: blob.type }));
  } catch {
    showError(t("err.demo"));
  }
});

const dropzone = el("dropzone");
["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("hot");
  }));
["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("hot");
  }));
dropzone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));

window.addEventListener("paste", (event) => {
  const wanted = state.mode === "video" ? "video/" : "image/";
  const item = [...(event.clipboardData?.items ?? [])].find((i) => i.type.startsWith(wanted));
  if (item) loadFile(item.getAsFile());
});

el("brush").addEventListener("input", (e) => (el("brushVal").textContent = e.target.value));
el("grow").addEventListener("input", (e) => (el("growVal").textContent = e.target.value));
el("crf").addEventListener("input", (e) => (el("crfVal").textContent = e.target.value));

document.querySelectorAll("#corners [data-corner]").forEach((button) =>
  button.addEventListener("click", () => {
    state.strokes.push({ rect: cornerRect(button.dataset.corner) });
    redrawOverlay();
  }));

el("undo").addEventListener("click", () => {
  state.strokes.pop();
  redrawOverlay();
});

el("clear").addEventListener("click", () => {
  state.strokes = [];
  redrawOverlay();
});

el("reset").addEventListener("click", () => setMode(state.mode));

el("run").addEventListener("click", async () => {
  el("run").disabled = true;
  hideError();
  const started = performance.now();
  try {
    if (state.mode === "video") await processVideo();
    else await inpaintImage();

    state.strokes = [];
    redrawOverlay();
    el("download").disabled = false;
    setStatus(`${t("st.done")} ${((performance.now() - started) / 1000).toFixed(1)} ${t("st.seconds")}`, 1);
  } catch (error) {
    console.error(error);
    showError(error.message || t("err.generic"));
    clearStatus();
  } finally {
    el("run").disabled = state.strokes.length === 0;
  }
});

el("download").addEventListener("click", () => {
  if (!state.resultBlob) return;
  const url = URL.createObjectURL(state.resultBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = state.resultName;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
});

/* ---------- language switching ------------------------------------------- */

function applyLanguage(lang) {
  state.lang = lang;
  document.documentElement.lang = lang;

  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    if (!node.dataset.en) node.dataset.en = node.innerHTML; // English is the source
    const translated = I18N[lang] && I18N[lang][key];
    node.innerHTML = translated ?? node.dataset.en;
  });

  document.querySelectorAll(".lang-btn").forEach((button) =>
    button.classList.toggle("active", button.dataset.lang === lang));

  // Strings the script sets itself need re-applying in the new language.
  el("pick").textContent = t(state.mode === "video" ? "pick.video" : "pick.image");
  if (el("editor").hidden) {
    el("hint").textContent = t(state.mode === "video" ? "hint.video" : "hint.image");
  }

  try {
    localStorage.setItem(LANG_KEY, lang);
  } catch {
    /* private mode: the choice just will not persist */
  }
}

document.querySelectorAll(".lang-btn").forEach((button) =>
  button.addEventListener("click", () => applyLanguage(button.dataset.lang)));

applyLanguage(currentLang());
setMode(state.mode);

/* WMR web — MI-GAN inpainting in the browser.
 *
 * Mirrors wmr/engines.py: crop around the mask, match the model aspect, resize,
 * inpaint, resize back, feather the seam. Including the halo fix — resampling smears
 * the watermark a pixel or two past its own edge, and any of that left outside the
 * hole gets read as valid context and pulled back into the fill.
 */

const MODEL_URL = "models/migan.onnx";
const ORT_DIR = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ORT_CPU = ORT_DIR + "ort.min.js";
const ORT_GPU = ORT_DIR + "ort.webgpu.min.js";
const MODEL_SIZE = 512;
const MARGIN_RATIO = 0.75;
const MIN_MARGIN = 64;
const HALO_DILATE = 2;
const FEATHER_PX = 1.6;

const el = (id) => document.getElementById(id);
const base = el("base");
const overlay = el("overlay");
const baseCtx = base.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlay.getContext("2d", { willReadFrequently: true });

const state = {
  image: null,
  strokes: [],
  current: null,
  drawing: false,
  session: null,
  loading: null,
  resultUrl: null,
};

/* ---------- small helpers ------------------------------------------------ */

function setStatus(text, fraction) {
  const box = el("status");
  box.hidden = false;
  el("statusText").textContent = text;
  el("barFill").style.width = `${Math.round((fraction ?? 0) * 100)}%`;
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

/* ---------- mask maths (ports of wmr/engines.py) -------------------------- */

/** Separable max filter: dilating a binary mask by `radius` pixels. */
function dilate(mask, w, h, radius) {
  if (radius <= 0) return mask;
  const pass = (src) => {
    const out = new Uint8Array(src.length);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        let hit = 0;
        const lo = Math.max(0, x - radius);
        const hi = Math.min(w - 1, x + radius);
        for (let i = lo; i <= hi && !hit; i++) hit = src[row + i];
        out[row + x] = hit;
      }
    }
    return out;
  };

  // Horizontal pass, then the same pass on the transpose = vertical pass.
  const transpose = (src, sw, sh) => {
    const out = new Uint8Array(src.length);
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) out[x * sh + y] = src[y * sw + x];
    }
    return out;
  };

  let work = pass(mask);
  work = transpose(work, w, h);
  [w, h] = [h, w];
  work = pass(work);
  return transpose(work, w, h);
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

  // Square it up so nothing gets squashed on the way into the model.
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

/* ---------- model -------------------------------------------------------- */

/** Pick the onnxruntime build to load.
 *
 * The WebGPU bundle ships a different wasm binary that is ~4x slower on the CPU path,
 * so loading it "just in case" penalises every visitor without a working GPU. Ask for
 * a real adapter first - `navigator.gpu` existing is not the same as WebGPU working.
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
    script.onerror = () => reject(new Error("onnxruntime-web gagal dimuat."));
    document.head.append(script);
  });

  ort.env.wasm.wasmPaths = ORT_DIR;
  ort.env.wasm.numThreads = 1; // threaded wasm needs COOP/COEP, which Pages cannot set
  ort.env.logLevel = "error";
  return gpu;
}

async function loadSession() {
  if (state.session) return state.session;
  if (state.loading) return state.loading;

  state.loading = (async () => {
    const gpu = await loadRuntime();

    setStatus("Mengunduh model MI-GAN…", 0);
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`Model gagal diunduh (HTTP ${response.status})`);

    const total = Number(response.headers.get("Content-Length")) || 0;
    const chunks = [];
    let received = 0;
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      setStatus(
        `Mengunduh model… ${(received / 1e6).toFixed(1)} MB`,
        total ? (received / total) * 0.9 : 0.5
      );
    }

    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }

    setStatus("Menyiapkan model…", 0.95);

    const backends = gpu ? ["webgpu", "wasm"] : ["wasm"];
    for (const backend of backends) {
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
    if (!state.session) throw new Error("Model tidak bisa dijalankan di browser ini.");

    el("engineNote").textContent =
      `MI-GAN berjalan di ${state.backend === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly)"}.`;
    return state.session;
  })();

  try {
    return await state.loading;
  } finally {
    state.loading = null;
  }
}

/* ---------- the pipeline ------------------------------------------------- */

async function inpaint() {
  const W = base.width;
  const H = base.height;

  // The overlay's alpha channel is the mask the user painted.
  const painted = overlayCtx.getImageData(0, 0, W, H).data;
  let mask = new Uint8Array(W * H);
  let any = false;
  for (let i = 0, p = 3; i < mask.length; i++, p += 4) {
    if (painted[p] > 8) {
      mask[i] = 1;
      any = true;
    }
  }
  if (!any) throw new Error("Belum ada area yang dicoret.");

  const grow = Number(el("grow").value);
  if (grow > 0) mask = dilate(mask, W, H, grow);

  const crop = cropWindow(maskBounds(mask, W, H), W, H);
  const session = await loadSession();
  setStatus("Menghapus watermark…", 0.4);
  await new Promise((r) => requestAnimationFrame(r));

  // Crop -> model resolution.
  const small = makeCanvas(MODEL_SIZE, MODEL_SIZE);
  const smallCtx = small.getContext("2d", { willReadFrequently: true });
  smallCtx.imageSmoothingQuality = "high";
  smallCtx.drawImage(base, crop.x, crop.y, crop.w, crop.h, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const smallPixels = smallCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;

  // The mask takes the same trip, but any partial coverage counts as hole and the
  // result is dilated: that is what keeps the resampling halo out of the context.
  const maskFull = makeCanvas(W, H);
  const maskFullCtx = maskFull.getContext("2d");
  const maskImage = maskFullCtx.createImageData(W, H);
  for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
    const v = mask[i] ? 255 : 0;
    maskImage.data[p] = v;
    maskImage.data[p + 1] = v;
    maskImage.data[p + 2] = v;
    maskImage.data[p + 3] = 255;
  }
  maskFullCtx.putImageData(maskImage, 0, 0);

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

  setStatus("Menyusun hasil…", 0.85);
  await new Promise((r) => requestAnimationFrame(r));

  // Model output -> back to crop resolution.
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

  // Feathered mask at crop resolution, so the patch has no hard edge.
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
}

/* ---------- painting ----------------------------------------------------- */

function redrawOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.strokeStyle = "rgba(255, 45, 85, 0.55)";
  overlayCtx.lineCap = "round";
  overlayCtx.lineJoin = "round";
  for (const stroke of state.strokes) {
    overlayCtx.lineWidth = stroke.size;
    overlayCtx.beginPath();
    stroke.points.forEach(([x, y], i) => {
      if (i === 0) overlayCtx.moveTo(x, y);
      else overlayCtx.lineTo(x, y);
    });
    if (stroke.points.length === 1) overlayCtx.lineTo(stroke.points[0][0] + 0.01, stroke.points[0][1]);
    overlayCtx.stroke();
  }
  const has = state.strokes.length > 0;
  el("undo").disabled = !has;
  el("clear").disabled = !has;
  el("run").disabled = !has;
}

function pointerPosition(event) {
  const rect = overlay.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) / rect.width) * overlay.width,
    ((event.clientY - rect.top) / rect.height) * overlay.height,
  ];
}

function brushSize() {
  // The slider is in display pixels; strokes live in image pixels.
  const rect = overlay.getBoundingClientRect();
  const scale = rect.width ? overlay.width / rect.width : 1;
  return Number(el("brush").value) * scale;
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

/* ---------- loading and wiring ------------------------------------------- */

function loadImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      state.image = img;
      state.strokes = [];
      base.width = overlay.width = img.naturalWidth;
      base.height = overlay.height = img.naturalHeight;
      baseCtx.drawImage(img, 0, 0);
      redrawOverlay();
      el("dropzone").hidden = true;
      el("editor").hidden = false;
      el("reset").hidden = false;
      el("download").disabled = true;
      hideError();
      clearStatus();
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

el("pick").addEventListener("click", () => el("file").click());
el("file").addEventListener("change", (event) => loadImage(event.target.files[0]));

el("demo").addEventListener("click", async () => {
  try {
    const response = await fetch("demo.jpg");
    loadImage(new File([await response.blob()], "demo.jpg", { type: "image/jpeg" }));
  } catch (error) {
    showError("Gambar contoh gagal dimuat.");
  }
});

const dropzone = el("dropzone");
["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("hot");
  })
);
["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("hot");
  })
);
dropzone.addEventListener("drop", (event) => loadImage(event.dataTransfer.files[0]));

window.addEventListener("paste", (event) => {
  const item = [...(event.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
  if (item) loadImage(item.getAsFile());
});

el("brush").addEventListener("input", (e) => (el("brushVal").textContent = e.target.value));
el("grow").addEventListener("input", (e) => (el("growVal").textContent = e.target.value));

el("undo").addEventListener("click", () => {
  state.strokes.pop();
  redrawOverlay();
});

el("clear").addEventListener("click", () => {
  state.strokes = [];
  redrawOverlay();
});

el("reset").addEventListener("click", () => {
  state.image = null;
  state.strokes = [];
  el("editor").hidden = true;
  el("dropzone").hidden = false;
  el("reset").hidden = true;
  el("download").disabled = true;
  el("file").value = "";
  clearStatus();
  hideError();
});

el("run").addEventListener("click", async () => {
  const button = el("run");
  button.disabled = true;
  hideError();
  try {
    await inpaint();
    state.strokes = [];
    redrawOverlay();
    el("download").disabled = false;
    setStatus("Selesai.", 1);
    setTimeout(clearStatus, 1800);
  } catch (error) {
    console.error(error);
    showError(error.message || "Gagal memproses gambar.");
    clearStatus();
  } finally {
    button.disabled = state.strokes.length === 0;
  }
});

el("download").addEventListener("click", () => {
  base.toBlob((blob) => {
    if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
    state.resultUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = state.resultUrl;
    link.download = "wmr-clean.png";
    link.click();
  }, "image/png");
});

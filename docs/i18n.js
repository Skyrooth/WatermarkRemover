/* Translations. The HTML carries English; `id` overrides it element by element.
 * Keys map to data-i18n attributes; a key ending in .html sets innerHTML so links
 * inside a sentence survive translation.
 */

const I18N = {
  en: {},

  id: {
    "lede.html":
      "Hapus watermark dari <strong>gambar</strong> dan <strong>video</strong> langsung " +
      "di browser. Tidak ada file yang di-upload — semuanya diproses di perangkat kamu.",
    "tab.image": "Gambar",
    "tab.video": "Video",
    "pick.image": "Pilih gambar",
    "pick.video": "Pilih video",
    "drop.hint": "atau seret ke sini · tempel dengan Ctrl+V · gambar dan video sama-sama bisa",
    "drop.demo": "Coba pakai contoh →",
    "hint.image": "Coret watermark-nya. Yang tertutup merah akan diganti.",
    "hint.video": "Coret watermark-nya. Mask ini dipakai untuk seluruh video.",
    "field.brush": "Ukuran kuas",
    "field.grow": "Perbesar mask",
    "field.growNote":
      "Kecil lebih baik — makin besar lubangnya, makin banyak yang harus ditebak.",
    "field.quality": "Kualitas encode",
    "field.qualityNote": "Makin kecil makin bagus, tapi filenya makin besar.",
    "corners.label": "Tandai pojok",
    "corners.note": "Veo dan Gemini menaruh mark-nya di pojok kanan bawah.",
    "why.paintFirst":
      "Coret dulu watermark-nya — tombolnya menyala begitu ada yang ditandai.",
    "btn.undo": "Undo",
    "btn.clear": "Bersihkan coretan",
    "btn.run": "Hapus watermark",
    "btn.download": "Download hasil",
    "btn.reset": "Mulai dengan file lain",
    "note.engines":
      "Gambar: MI-GAN (30 MB). Video: ffmpeg (32 MB). Diunduh sekali lalu di-cache browser.",

    "h.how": "Cara kerjanya",
    "p.how":
      "Watermark tidak bisa \"dibalik\" kecuali logonya sudah diketahui persis. Yang bisa " +
      "dilakukan adalah membuang area yang kamu tandai lalu mengisinya kembali dari " +
      "piksel di sekitarnya.",
    "li.image.html":
      "<strong>Gambar</strong> — <em>inpainting</em> dengan model <strong>MI-GAN</strong> " +
      "lewat <code>onnxruntime-web</code>. Sekitar 1,6 detik per gambar.",
    "li.video.html":
      "<strong>Video</strong> — filter <strong><code>delogo</code></strong> milik ffmpeg, " +
      "dijalankan sebagai WebAssembly. Tiap frame diinterpolasi dari tepi kotak yang kamu " +
      "tandai. Audio aslinya disalin apa adanya, tanpa encode ulang.",
    "p.veo.html":
      "Untuk watermark kecil semi-transparan yang posisinya tetap — persis seperti " +
      "watermark Veo/Gemini — <code>delogo</code> sudah cukup dan sangat cepat. Kalau di " +
      "belakang watermark ada tekstur atau detail, pakai versi desktop dengan engine " +
      "<code>migan</code>: hasilnya jelas lebih rapi.",

    "h.limits": "Batasannya",
    "li.limit.threads.html":
      "Video diproses satu thread. GitHub Pages tidak bisa mengirim header " +
      "<code>COOP/COEP</code> yang dibutuhkan ffmpeg versi multi-thread, jadi klip panjang " +
      "akan lama. Untuk klip panjang, pakai versi desktop.",
    "li.limit.moving.html":
      "Watermark video yang <strong>berpindah posisi</strong> hanya ditangani versi " +
      "desktop, yang melacaknya per frame.",
    "li.limit.synthid.html":
      "Yang dihapus hanya watermark <strong>visual</strong>. Watermark tak-kasat-mata " +
      "seperti SynthID tidak disentuh.",

    "h.desktop": "Versi desktop",
    "p.desktop.html":
      "Lebih cepat dan lebih lengkap: engine <strong>LaMa</strong> untuk kualitas " +
      "tertinggi, <strong>MI-GAN di GPU</strong> untuk video (klip 10 detik selesai dalam " +
      "29 detik), dan pelacakan watermark bergerak. Ada di " +
      "<a href=\"https://github.com/Skyrooth/WatermarkRemover\">github.com/Skyrooth/WatermarkRemover</a>.",

    "p.legal": "Pakai untuk konten milik sendiri atau yang kamu punya izinnya.",
    "footer.html": "MIT · <a href=\"https://github.com/Skyrooth/WatermarkRemover\">Kode sumber</a>",

    "err.noStrokes": "Belum ada area yang dicoret.",
    "err.tooSmall": "Coretannya terlalu kecil untuk dikenali.",
    "err.pickImage": "Pilih file gambar.",
    "err.pickVideo": "Pilih file video.",
    "err.pickFile": "Pilih file gambar atau video.",
    "err.badImage": "Gambar tidak bisa dibaca.",
    "err.badVideo": "Video tidak bisa dibaca. Coba MP4 (H.264).",
    "err.demo": "Contoh gagal dimuat.",
    "err.model": "Model gagal diunduh",
    "err.noBackend": "Model tidak bisa dijalankan di browser ini.",
    "err.ort": "onnxruntime-web gagal dimuat.",
    "err.ffmpegScript": "ffmpeg.js gagal dimuat.",
    "err.ffmpegLoad": "ffmpeg tidak selesai dimuat. Coba muat ulang halaman.",
    "err.ffmpegRun": "ffmpeg gagal. Coba format video lain.",
    "err.generic": "Gagal memproses file.",
    "err.delogoMoving": "Coretannya kosong.",

    "st.downloadModel": "Mengunduh model MI-GAN…",
    "st.downloading": "Mengunduh model…",
    "st.prepModel": "Menyiapkan model…",
    "st.removing": "Menghapus watermark…",
    "st.composing": "Menyusun hasil…",
    "st.loadFfmpeg": "Memuat ffmpeg…",
    "st.prepFfmpeg": "Menyiapkan ffmpeg (32 MB)…",
    "st.prepVideo": "Menyiapkan video…",
    "st.processing": "Memproses video…",
    "st.done": "Selesai dalam",
    "st.seconds": "detik.",
    "st.videoInfo": "detik",
    "st.maskAllFrames": "Mask ini dipakai untuk seluruh video",
    "st.previewFrame": "Preview ini frame detik",
  },
};

const EN_FALLBACK = {
  "err.noStrokes": "Nothing is painted yet.",
  "err.tooSmall": "That stroke is too small to detect.",
  "err.pickImage": "Choose an image file.",
  "err.pickVideo": "Choose a video file.",
  "err.pickFile": "Choose an image or a video file.",
  "err.badImage": "That image could not be read.",
  "err.badVideo": "That video could not be read. Try MP4 (H.264).",
  "err.demo": "The example failed to load.",
  "err.model": "Model download failed",
  "err.noBackend": "The model cannot run in this browser.",
  "err.ort": "onnxruntime-web failed to load.",
  "err.ffmpegScript": "ffmpeg.js failed to load.",
  "err.ffmpegLoad": "ffmpeg never finished loading. Try reloading the page.",
  "err.ffmpegRun": "ffmpeg failed. Try a different video format.",
  "err.generic": "Could not process that file.",
  "err.delogoMoving": "Nothing is painted yet.",

  "corners.label": "Mark a corner",
  "corners.note": "Veo and Gemini put their mark in the bottom-right corner.",
  "why.paintFirst":
    "Paint over the watermark first — the button turns on once something is marked.",

  "pick.image": "Choose image",
  "pick.video": "Choose video",
  "hint.image": "Paint over the watermark. Anything covered in red gets replaced.",
  "hint.video": "Paint over the watermark. This mask is used for the whole video.",

  "st.downloadModel": "Downloading the MI-GAN model…",
  "st.downloading": "Downloading model…",
  "st.prepModel": "Preparing the model…",
  "st.removing": "Removing the watermark…",
  "st.composing": "Composing the result…",
  "st.loadFfmpeg": "Loading ffmpeg…",
  "st.prepFfmpeg": "Preparing ffmpeg (32 MB)…",
  "st.prepVideo": "Preparing the video…",
  "st.processing": "Processing video…",
  "st.done": "Done in",
  "st.seconds": "seconds.",
  "st.videoInfo": "seconds",
  "st.maskAllFrames": "This mask is used for the whole video",
  "st.previewFrame": "Preview is the frame at",
};

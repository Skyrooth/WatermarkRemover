# WMR — Watermark Remover (gambar + video)

> 🇬🇧 [English version](README.md)

Tool lokal untuk menghapus watermark / logo dari **gambar** dan **video**.
Kamu tandai area watermark-nya (brush atau koordinat box), tool-nya meng-*inpaint*
area itu. Semua proses jalan di PC ini — tidak ada file yang di-upload ke mana pun.

> **Versi web (gambar saja): [skyrooth.github.io/WatermarkRemover](https://skyrooth.github.io/WatermarkRemover/)**
> Tanpa install apa pun. MI-GAN berjalan di browser kamu lewat `onnxruntime-web`,
> ~1,6 detik per gambar. Untuk **video** dan engine LaMa, pakai versi desktop di bawah.

## Kenapa bukan fork dari gemini-watermark-remover

Repo `GargantuaX/gemini-watermark-remover` pakai *reverse alpha blending* dengan
**alpha map logo Gemini yang sudah di-embed**. Rumusnya membalik
`hasil = asli×(1−α) + logo×α`, dan itu presisi **hanya karena α + warna logo sudah
diketahui persis**. Untuk logo sembarang (TikTok, CapCut, Sora, stock photo, dll)
α-nya tidak diketahui, jadi seluruh core Gemini-specific-nya tidak bisa dipakai.
Pendekatan yang benar untuk watermark umum adalah **mask + inpainting** — itu yang
dipakai di sini.

## Install

Sudah siap pakai (`uv sync` sudah dijalankan). Kalau pindah PC:

```bash
uv sync
```

Butuh **ffmpeg + ffprobe** di PATH (sudah ada: ffmpeg 8.1.2 full build).

## GUI

```bash
uv run python app.py
```

Buka http://localhost:7861

- **Tab Gambar** — upload gambar → coret watermark pakai brush → pilih engine → *Hapus watermark*
- **Tab Video** — upload video → frame preview muncul otomatis → coret watermark di frame itu
  (mask berlaku untuk seluruh video) → *Hapus watermark*. Audio asli ikut, tidak di-encode ulang.

## CLI

```bash
# gambar
uv run wmr image foto.png --box 940,590,300,90 -o bersih.png   # engine lama (default)

# koordinat boleh persen
uv run wmr image foto.png --box "72%,82%,24%,13%" --engine ns

# pakai file mask (putih = dihapus)
uv run wmr image foto.png --mask mask.png --engine lama

# video (default: migan di GPU)
uv run wmr video klip.mp4 --box 940,590,300,90 -o bersih.mp4

# video (paling cepat, ffmpeg delogo)
uv run wmr video klip.mp4 --box 940,590,300,90 --engine delogo

# watermark BERGERAK - dilacak otomatis di tiap frame
uv run wmr video klip.mp4 --box 940,590,300,90 --track

# kalau kotaknya kamu ambil dari detik 3, bukan awal video
uv run wmr video klip.mp4 --box 940,590,300,90 --track --at 3

# info file
uv run wmr probe klip.mp4
```

Flag penting:

| Flag | Arti |
|---|---|
| `--box X,Y,W,H` | Area watermark. Boleh diulang untuk beberapa watermark. Boleh persen. |
| `--mask file.png` | Alternatif box: mask gambar, putih = dihapus. |
| `--engine` | gambar: `lama` (default) · video: `migan` (default) · `delogo` (video saja) · `telea` · `ns` |
| `--grow N` | Melebarkan mask N px sebelum inpaint (default 3). Naikkan kalau masih ada sisa tepi. |
| `--crf N` | Kualitas encode video, kecil = lebih bagus (default 18). |
| `--track` | Lacak watermark yang bergerak (video). |
| `--at S` | Detik di mana `--box`/`--mask` diambil (default 0). |
| `--json` | Output hasil sebagai JSON. |

## Engine

Diukur di PC ini (720p, GTX 1660 SUPER, `scripts/bench_engines.py`):

| Engine | Perangkat | Kecepatan | Cocok untuk |
|---|---|---|---|
| `lama` | CPU saja | 2,57 detik/frame | **Default untuk gambar.** Kualitas tertinggi, konsisten. |
| `migan` | **GPU (DirectML)** | **0,091 detik/frame** | **Default untuk video.** AI inpainting yang sanggup real-time-ish. |
| `delogo` | CPU (ffmpeg) | real-time | Logo kecil semi-transparan, posisi tetap. Tidak bisa `--track`. |
| `telea` / `ns` | CPU | 0,055 detik/frame | Cepat, dan kuat justru di background halus. |

Video 6 detik (180 frame, 720p): `migan` **26 detik**, `lama` **7,6 menit**.

### Sisa watermark yang terukur

`scripts/eval_removal.py` membandingkan hasil dengan klip referensi bersih dan
melaporkan sisa error di area watermark (makin kecil makin baik):

| Engine | Background halus | Background bertekstur | Waktu (180 frame) |
|---|---|---|---|
| `lama` | **21,4%** | **24,9%** | 457 detik |
| `telea` | 40,1% | 34,8% | 19 detik |
| `ns` | 39,8% | 35,0% | 18 detik |
| `migan` | 48,0% | 42,9% | 26 detik |

**Baca ini dengan hati-hati.** `lama` menang di mana-mana — itu solid. Tapi
`telea`/`ns` mengalahkan `migan` di sini **karena kedua background uji ini sintetis
dan mulus secara lokal**, dan itu justru medan terbaik untuk difusi. Di konten nyata
yang punya struktur (garis, tekstur, objek), difusi menghasilkan smear sedangkan
inpainting AI merekonstruksi — terlihat jelas di `samples/compare_engines.png` pada
pola checkerboard. Jadi angka di atas **tidak** menyelesaikan perdebatan
`migan` vs `telea` untuk footage asli; coba keduanya di klip kamu sendiri.

Yang bisa disimpulkan dari angka ini: `lama` paling akurat, dan `--grow` kecil lebih
baik daripada besar.

### Model

Keduanya sudah terunduh ke `models/`. Kalau setup ulang:

```bash
uv add onnxruntime-directml
uv run wmr download-model --model migan   # 30 MB
uv run wmr download-model --model lama    # 208 MB
```

**Kenapa `migan` jalan di GPU tapi `lama` tidak.** LaMa memakai *fast Fourier
convolution*, dan DirectML tidak punya kernel untuk op FFT-nya — graph-nya ke-load
lalu mati saat runtime. MI-GAN tidak pakai FFT jadi mulus di DirectML. Setiap engine
ONNX **menguji provider-nya dengan satu forward pass** saat dibuat dan otomatis turun
ke CPU kalau gagal (lihat `OnnxInpaintEngine.fallback_from`) — jadi tidak ada crash
di tengah proses video. Untuk LaMa di GPU perlu jalur CUDA (`onnxruntime-gpu` +
CUDA 12 + cuDNN 9), belum terpasang di PC ini.

Perbedaan lain yang ditangani otomatis: polaritas mask **terbalik** (LaMa 255 = lubang,
MI-GAN 255 = dipertahankan), dan LaMa input-nya fixed 512×512 sedangkan MI-GAN dinamis.

Benchmark ulang kapan saja:

```bash
uv run python scripts/bench_engines.py
```

## Struktur

```
wmr/
  ffmpeg.py     probe, pipe frame raw, mux audio
  mask.py       box (absolut/persen), file mask, dilate, bbox, connected components
  engines.py    OpenCV (telea/ns) + OnnxInpaintEngine → MI-GAN & LaMa
  tracking.py   pelacak template untuk watermark bergerak
  image_ops.py  baca/tulis gambar (alpha dipertahankan) + pipeline gambar
  video_ops.py  backend delogo, per-frame, dan tracked
  download.py   unduh model MI-GAN / LaMa
  cli.py        CLI wmr
app.py          GUI Gradio
scripts/        make_samples, eval_tracking, eval_removal, bench_engines, probe_migan
tests/          29 test pytest
models/         migan.onnx + lama_fp32.onnx (tidak masuk git)
samples/        corpus uji + ground truth (tidak masuk git)
```

### Detail implementasi yang penting

`OnnxInpaintEngine` tidak mengirim seluruh frame ke model. Alurnya: ambil **crop di
sekitar mask** (konteks secukupnya, bukan 4K penuh) → lebarkan ke aspect ratio model →
resize ke resolusi model → inpaint → resize balik → **blend dengan tepi di-feather**
supaya sambungannya tidak kelihatan.

Satu jebakan yang sudah diperbaiki: saat crop di-resize, watermark-nya "meleber" 1–2
piksel melewati tepinya sendiri. Kalau mask di-resize dengan NEAREST, halo itu berada
di **luar** lubang, model membacanya sebagai konteks sah dan menariknya kembali ke
hasil isian. Efeknya besar — error uji 45,9 vs 1,3. Karena itu mask di-resize dengan
INTER_LINEAR (coverage sebagian dihitung sebagai lubang) lalu di-dilate 2 px.

## Test

```bash
uv run --with pytest pytest -q
```

## Watermark bergerak (`--track`)

Kalau logonya pindah posisi atau loncat antar sudut, satu mask statis tidak cukup.
Dengan `--track` (atau centang **"Watermark bergerak"** di GUI) tool ini jalan dua pass:

1. **Lacak** — template matching pada *gradient magnitude*, bukan warna mentah.
   Tepi logo tetap sama walau background di balik overlay semi-transparan berubah.
   Tiap frame dicari dulu di sekitar posisi sebelumnya (murah); kalau skornya jatuh,
   barulah sapu satu frame penuh — itu yang menangkap lompatan antar sudut.
2. **Hapus** — mask dipindah ke posisi hasil pelacakan di setiap frame.

Frame yang skor kecocokannya rendah tidak ditebak asal: posisinya **diinterpolasi**
dari frame yang yakin di kiri-kanannya.

**Akurasi terukur.** `samples/truth.json` menyimpan posisi logo yang sebenarnya di
tiap frame (karena sample-nya dibuat sendiri oleh `scripts/make_samples.py`), jadi
errornya angka nyata, bukan kesan:

```
uv run python scripts/eval_tracking.py
→ mean 0.00px   median 0.00px   p95 0.00px   max 0.00px   (180 frame, 2 klip)
```

Satu jebakan yang sudah diperbaiki: median filter untuk meredam jitter **memangkas
titik balik**. Saat logo memantul, median dari `[8, 1, 8]` adalah `8` — puncak
pantulannya hilang dan mask jadi tertinggal di frame itu. Sekarang median hanya
dipakai untuk *menolak outlier* (kalau simpangannya melebihi skala gerak klip itu
sendiri), bukan mengganti semua posisi. Itu yang membawa error dari 0,12 px ke 0,00 px.

`--engine delogo` tidak bisa dipakai bersama `--track` — filter ffmpeg itu hanya
menggambar satu kotak tetap untuk seluruh klip. Tool-nya menolak dengan pesan jelas.

## Test dan evaluasi

```bash
uv run --with pytest pytest -q                    # 29 test
uv run python scripts/make_samples.py             # regenerasi corpus uji
uv run python scripts/eval_tracking.py            # akurasi pelacakan (px)
uv run python scripts/eval_removal.py             # sisa watermark vs referensi bersih
uv run python scripts/bench_engines.py            # kecepatan tiap engine
```

Corpus ujinya dibuat di numpy, bukan pakai sumber sintetis ffmpeg — `gradients` acak
tiap run, dan `testsrc2` adalah kasus terburuk untuk inpainting. Yang lebih penting:
tiap klip ber-watermark punya **kembaran bersih yang identik piksel per piksel**, jadi
"sisa watermark" bisa diukur sebagai angka. Ada dua background — sinus halus dan
noise fraktal — karena background halus memihak difusi dan FFT-nya LaMa, dan bias
seperti itu bisa membuat kita memilih default yang salah.

## Catatan

- Yang dihapus hanya watermark **visual**. Watermark tak-kasat-mata (mis. SynthID
  milik Google) tidak disentuh.
- Tandai watermark **seketat mungkin**. Makin besar lubangnya makin banyak yang harus
  ditebak model — menaikkan `--grow` dari 2 ke 8 justru memperburuk hasil di semua engine.
- Pakai untuk konten milik sendiri atau yang kamu punya izinnya.

# lvglimg

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/Junbo-Zheng/lvglimg/actions/workflows/ci.yml/badge.svg)](https://github.com/Junbo-Zheng/lvglimg/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lvglimg.svg)](https://pypi.org/project/lvglimg/)

Decode LVGL `.bin` image files to PNG — including **v8 + v9** headers and
**NONE / RLE / LZ4 / LZ4_HC** compression.

LVGL's own `LVGLImage.py` is a one-way encoder: it can *write* compressed bins,
but `from_bin` ignores the flags field and cannot decompress them back. `lvglimg`
fills that gap, producing pixel-accurate PNGs from real device asset bins.

- **Self-contained** — depends only on Pillow + lz4. No LVGL source tree, no
  `LVGLImage.py` on disk.
- **All v9 color formats** — L8, indexed (I1/2/4/8), alpha-only (A1/2/4/8),
  RGB565, RGB888, ARGB8888, XRGB8888, ARGB8565, RGB565A8.
- **v8 true-color** — `TRUE_COLOR` (cf=4, RGB565 at the display's native depth) / `TRUE_COLOR_ALPHA` (cf=5, BGRA).
- **CLI + library** — use it on the command line or import the decoder.

## Install

```bash
pip install lvglimg
```

## CLI

```bash
# single file → sibling .png
lvglimg image.bin

# explicit output path
lvglimg image.bin out.png

# batch a directory (recursive) → out/<same subdirs>/*.png
lvglimg assets/ out/
# default: a sibling '<input>_preview/' dir next to the input
lvglimg assets/       # → assets_preview/ (sibling of assets/)
# finds assets/*.bin AND assets/**/<sub>/*.bin; mirrors the subdir layout
# in the output dir (avoids name collisions across folders)
# viewable images in the tree (.jpg/.jpeg/.png/.gif/.bmp/.webp) are copied
# to the output as-is — no decoding, the preview dir holds the full asset set
```

### Headerless raw buffers

Some vendor assets are headerless raw BGRA pixel dumps with no LVGL
header. When normal decoding fails, `lvglimg` retries
automatically with a size guessed by a row-coherence scan (a wrong width
misaligns rows into noise; the correct one keeps image structure smooth)
and prints a note so the guess is visible:

```
$ lvglimg raw/
    [raw auto-detected 64x48 BGRA: raw/boot.bin]
OK  raw/boot.bin -> ./raw_preview/boot.png
```

To skip the guess, pass the size explicitly (or force raw mode on a file
that happens to parse as v8):

```bash
lvglimg --raw 64x48 asset.bin
lvglimg --raw auto asset.bin
```

Size auto-detection needs `numpy` — install it with `pip install
lvglimg[auto]` (or `pip install -e ".[dev]"`, which includes it). Everything
else works without it.

```
$ lvglimg --version
lvglimg 0.0.4
```

## Library

```python
from lvglimg.decoder import decode, decode_file

# from bytes
img = decode(bin_bytes)
img.save("out.png")

# from a file path
img = decode_file("image.bin")
```

`decode()` returns a `PIL.Image.Image` (mode `RGBA`, `RGB`, or `L` depending on
the source format), so you can inspect, transform, or re-encode it however you
like.

## How it works

The LVGL v9 on-disk layout is a 12-byte header followed by a body that may be a
compress block:

```
header: magic(0x19) cf flags w h stride reserved
body  : [method u32][clen u32][raw_len u32][payload]   # iff flags & 0x08
```

`lvglimg` reads the `flags` field to detect compression (the same bit LVGL sets
when encoding), decompresses RLE / LZ4 / LZ4_HC, then unpacks pixels to RGBA
using the same channel layout and `bit_extend` upscaling LVGL renders on-device —
so output matches the display. v8 files (no magic byte) are dispatched to the
4-byte bitfield header path.

> **Note** — v8 support covers `TRUE_COLOR` (cf=4, RGB565 at the display's
> native depth) and `TRUE_COLOR_ALPHA` (cf=5, 4 B/px BGRA), the layouts real
> device asset bins use. Other legacy v8 formats raise a clear error rather than
> silently producing garbage.

## Development

```bash
pip install -e ".[dev]"
pytest            # 21 tests, runs against src/ with no install
ruff check src tests main.py
mypy
./main.py image.bin   # run from source, no install
```

### Local pre-push checks

Before pushing, run `git pre` — it executes `.githooks/pre-push`
(ruff + mypy + pytest on Python 3.10/3.11/3.12, mirroring CI).
Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

## License

This project is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) or <https://www.apache.org/licenses/LICENSE-2.0> for the
full text.

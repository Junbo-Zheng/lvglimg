# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Junbo Zheng
"""LVGL ``.bin`` image decoder (v8 + v9) → :class:`PIL.Image.Image`.

Self-contained: depends only on Pillow + lz4. The v9 color-format unpacking
mirrors LVGL's own ``LVGLImage`` semantics (palette byte order, RGB565A8 split
alpha, ARGB8565 layout, bit_extend upscaling) so output matches what LVGL
renders on-device. Compression (RLE / LZ4 / LZ4_HC) is decoded in-place — the
official ``LVGLImage.from_bin`` skips decompression, which is the gap this
module fills.

The LVGL v9 on-disk layout (little-endian):

    +0  u8   magic = 0x19
    +1  u8   cf (low 5 bits) | reserved
    +2  u16  flags (bit 0x08 = compressed)
    +4  u16  width
    +6  u16  height
    +8  u16  stride (row length in bytes, incl. padding)
    +10 u16  reserved
    +12 ..   body (palette? + bitmap, or compress block + payload)

The compress block (present iff flags & 0x08):

    +0  u32  method   (1 = RLE, 2 = LZ4 / LZ4_HC — wire-identical)
    +4  u32  clen     compressed length
    +8  u32  raw_len  uncompressed length
    +12 ..  payload
"""

from __future__ import annotations

import struct
from enum import IntEnum

import lz4.block
from PIL import Image

MAGIC_V9 = 0x19
FLAG_COMPRESSED = 0x08

# v8 header is a 4-byte bitfield (no magic):
#   cf:5 | reserved:3 | w:11 | h:11   (little-endian u32)
V8_HEADER_LEN = 4
# v8 true-color color formats (lv_img_cf_t):
#   cf=4 TRUE_COLOR       → pixels at the display's native color depth. On an
#                           RGB565 display that is 2 B/px little-endian (no alpha).
#   cf=5 TRUE_COLOR_ALPHA → 4 B/px BGRA (native color + 1 alpha byte).
V8_CF_TRUE_COLOR = 4  # LV_IMG_CF_TRUE_COLOR
V8_CF_TRUE_COLOR_ALPHA = 5  # LV_IMG_CF_TRUE_COLOR_ALPHA

# Compress method wire values (match LVGL CompressMethod).
COMPRESS_RLE = 1
COMPRESS_LZ4 = 2  # LZ4 and LZ4_HC share this wire value


class ColorFormat(IntEnum):
    """LVGL v9 color format enum (``lv_color_format_t``)."""

    UNKNOWN = 0x00
    L8 = 0x06
    I1 = 0x07
    I2 = 0x08
    I4 = 0x09
    I8 = 0x0A
    A1 = 0x0B
    A2 = 0x0C
    A4 = 0x0D
    A8 = 0x0E
    RGB888 = 0x0F
    ARGB8888 = 0x10
    XRGB8888 = 0x11
    RGB565 = 0x12
    ARGB8565 = 0x13
    RGB565A8 = 0x14

    @property
    def bpp(self) -> int:
        """Bits per pixel."""
        return {
            ColorFormat.UNKNOWN: 0,
            ColorFormat.L8: 8,
            ColorFormat.I1: 1,
            ColorFormat.I2: 2,
            ColorFormat.I4: 4,
            ColorFormat.I8: 8,
            ColorFormat.A1: 1,
            ColorFormat.A2: 2,
            ColorFormat.A4: 4,
            ColorFormat.A8: 8,
            ColorFormat.ARGB8888: 32,
            ColorFormat.XRGB8888: 32,
            ColorFormat.RGB565: 16,
            ColorFormat.RGB565A8: 16,
            ColorFormat.ARGB8565: 24,
            ColorFormat.RGB888: 24,
        }[self]

    @property
    def ncolors(self) -> int:
        """Palette entries for indexed formats, else 0."""
        return {
            ColorFormat.I1: 2,
            ColorFormat.I2: 4,
            ColorFormat.I4: 16,
            ColorFormat.I8: 256,
        }.get(self, 0)

    @property
    def is_indexed(self) -> bool:
        return self.ncolors != 0

    @property
    def is_alpha_only(self) -> bool:
        return ColorFormat.A1 <= self <= ColorFormat.A8


class LvglImageError(Exception):
    """Raised when an LVGL .bin cannot be decoded."""


def _bit_extend(value: int, bpp: int) -> int:
    """Extend a ``bpp``-bit channel to 8 bits (LVGL's interpolation, reduces
    rounding error). Mirrors ``LVGLImage.bit_extend``."""
    if value == 0:
        return 0
    res = value
    n = bpp
    while n < 8:
        res |= value << (8 - n)
        n += bpp
    return res & 0xFF


def _destride(body: bytes, stride: int, row_bytes: int, h: int) -> bytes:
    """Strip per-row stride padding, returning ``row_bytes * h`` compact bytes."""
    if stride == row_bytes:
        return body[: row_bytes * h]
    out = bytearray()
    for y in range(h):
        out += body[y * stride : y * stride + row_bytes]
    return bytes(out)


def _rle_decompress(comp: bytes, blksize: int) -> bytes:
    """Reverse of ``RLEImage.rle_compress``.

    Control byte high bit set  → literal run:  cnt = ctrl & 0x7f, copy
    ``cnt * blksize`` bytes verbatim.
    Control byte high bit clear → repeat run:  cnt = ctrl, read one ``blksize``
    block and emit it ``cnt`` times.
    """
    out = bytearray()
    i = 0
    n = len(comp)
    while i < n:
        ctrl = comp[i]
        i += 1
        if ctrl & 0x80:
            cnt = ctrl & 0x7F
            out += comp[i : i + cnt * blksize]
            i += cnt * blksize
        else:
            cnt = ctrl
            blk = comp[i : i + blksize]
            i += blksize
            out += blk * cnt
    return bytes(out)


def _decode_v9_header(d: bytes) -> tuple[ColorFormat, int, int, int, int]:
    """Return (cf, w, h, stride, flags)."""
    if len(d) < 12:
        raise LvglImageError(f"v9 header truncated: {len(d)} bytes")
    cf_val = d[1] & 0x1F
    try:
        cf = ColorFormat(cf_val)
    except ValueError as exc:
        raise LvglImageError(f"unknown v9 color format: 0x{cf_val:02x}") from exc
    flags = int.from_bytes(d[2:4], "little")
    w = int.from_bytes(d[4:6], "little")
    h = int.from_bytes(d[6:8], "little")
    stride = int.from_bytes(d[8:10], "little")
    return cf, w, h, stride, flags


def _maybe_decompress(body: bytes, cf: ColorFormat, flags: int) -> bytes:
    """If the compressed flag is set, decode the compress block."""
    if not (flags & FLAG_COMPRESSED):
        return body
    if len(body) < 12:
        raise LvglImageError("compressed body too short for compress header")
    method = int.from_bytes(body[0:4], "little")
    clen = int.from_bytes(body[4:8], "little")
    raw_len = int.from_bytes(body[8:12], "little")
    payload = body[12 : 12 + clen]
    if method == COMPRESS_LZ4:
        return bytes(lz4.block.decompress(payload, uncompressed_size=raw_len))
    if method == COMPRESS_RLE:
        blksize = max(1, (cf.bpp + 7) // 8)
        return _rle_decompress(payload, blksize)
    raise LvglImageError(f"unknown compress method: {method}")


def _unpack_alpha_bits(data: bytes, bpp: int, w: int, h: int) -> bytes:
    """Unpack 1/2/4-bit alpha-only rows to one byte per pixel (0..255)."""
    if bpp == 8:
        return data
    # LVGL gamma: A1→{0,255}, A2→{0,85,170,255}, A4→{0,17,...,255}.
    gamma = [round(i * 255 / ((1 << bpp) - 1)) for i in range(1 << bpp)]
    out = bytearray()
    row_bytes = (w * bpp + 7) // 8
    for y in range(h):
        row = data[y * row_bytes : y * row_bytes + row_bytes]
        for x in range(w):
            shift = 8 - bpp - (x * bpp % 8)
            idx = (row[x * bpp // 8] >> shift) & ((1 << bpp) - 1)
            out.append(gamma[idx])
    return bytes(out)


def _unpack_index_bits(data: bytes, bpp: int, w: int, h: int) -> bytes:
    """Unpack 1/2/4-bit index rows to one byte per pixel (index value)."""
    if bpp == 8:
        return data
    out = bytearray()
    row_bytes = (w * bpp + 7) // 8
    mask = (1 << bpp) - 1
    for y in range(h):
        row = data[y * row_bytes : y * row_bytes + row_bytes]
        for x in range(w):
            shift = 8 - bpp - (x * bpp % 8)
            out.append((row[x * bpp // 8] >> shift) & mask)
    return bytes(out)


def _palette_to_rgba(palette: bytes, ncolors: int) -> list[tuple[int, int, int, int]]:
    """LVGL palette is BGRA per entry → RGBA tuples."""
    return [
        (
            palette[i * 4 + 2],  # R
            palette[i * 4 + 1],  # G
            palette[i * 4 + 0],  # B
            palette[i * 4 + 3],  # A
        )
        for i in range(ncolors)
    ]


def _rgb565_to_rgb(pixels: bytes, count: int) -> bytes:
    """Unpack little-endian RGB565 pairs → flat R,G,B bytes (bit_extend)."""
    out = bytearray()
    for i in range(count):
        p = pixels[2 * i] | (pixels[2 * i + 1] << 8)
        out.append(_bit_extend((p >> 11) & 0x1F, 5))  # R
        out.append(_bit_extend((p >> 5) & 0x3F, 6))  # G
        out.append(_bit_extend(p & 0x1F, 5))  # B
    return bytes(out)


def _decode_v9_body(
    cf: ColorFormat, w: int, h: int, stride: int, body: bytes
) -> Image.Image:
    count = w * h
    # stride == 0 is valid LVGL: means "default stride, 1-byte aligned" =
    # (w * bpp + 7) // 8 bytes per row. Without this, _destride would repeat
    # row 0 for every row (stride 0 != row_bytes -> body[0:row_bytes] each row).
    if stride == 0:
        stride = (w * cf.bpp + 7) // 8

    if cf.is_indexed:
        pal_bytes = b"".join(bytes(p) for p in _palette_to_rgba(body, cf.ncolors))
        bm = body[cf.ncolors * 4 :]
        row_bytes = (w * cf.bpp + 7) // 8
        bm = _destride(bm, stride, row_bytes, h)
        idx = _unpack_index_bits(bm, cf.bpp, w, h)
        rgba = bytearray()
        for i in idx:
            rgba += pal_bytes[i * 4 : i * 4 + 4]
        return Image.frombytes("RGBA", (w, h), bytes(rgba))

    if cf.is_alpha_only:
        row_bytes = (w * cf.bpp + 7) // 8
        alpha = _unpack_alpha_bits(_destride(body, stride, row_bytes, h), cf.bpp, w, h)
        rgba = bytearray()
        for a in alpha:
            rgba += b"\x00\x00\x00" + bytes([a])
        return Image.frombytes("RGBA", (w, h), bytes(rgba))

    if cf is ColorFormat.L8:
        return Image.frombytes("L", (w, h), _destride(body, stride, w, h))

    if cf is ColorFormat.RGB565:
        rgb = _rgb565_to_rgb(_destride(body, stride, w * 2, h), count)
        return Image.frombytes("RGB", (w, h), rgb)

    if cf is ColorFormat.RGB888:
        # LVGL stores B,G,R per pixel.
        return Image.frombytes(
            "RGB", (w, h), _destride(body, stride, w * 3, h), "raw", "BGR"
        )

    if cf in (ColorFormat.ARGB8888, ColorFormat.XRGB8888):
        # LVGL stores B,G,R,A per pixel.
        return Image.frombytes(
            "RGBA", (w, h), _destride(body, stride, w * 4, h), "raw", "BGRA"
        )

    if cf is ColorFormat.ARGB8565:
        # 3 bytes/pixel: low, high (of RGB565), alpha.
        flat = _destride(body, stride, w * 3, h)
        rgba = bytearray()
        for i in range(count):
            p = flat[3 * i] | (flat[3 * i + 1] << 8)
            rgba.append(_bit_extend((p >> 11) & 0x1F, 5))  # R
            rgba.append(_bit_extend((p >> 5) & 0x3F, 6))  # G
            rgba.append(_bit_extend(p & 0x1F, 5))  # B
            rgba.append(flat[3 * i + 2])  # A
        return Image.frombytes("RGBA", (w, h), bytes(rgba))

    if cf is ColorFormat.RGB565A8:
        # Color stride = stride; alpha stride = stride // 2 (LVGL data_len rule).
        color = _destride(body, stride, w * 2, h)
        a_stride = stride // 2 if stride else w
        alpha = _destride(body[stride * h :], a_stride, w, h)
        rgb = _rgb565_to_rgb(color, count)
        rgba = bytearray()
        for i in range(count):
            rgba += rgb[3 * i : 3 * i + 3] + bytes([alpha[i]])
        return Image.frombytes("RGBA", (w, h), bytes(rgba))

    raise LvglImageError(f"unsupported v9 color format: {cf.name}")


def _decode_v8(d: bytes) -> Image.Image:
    """Decode an LVGL v8 image.

    v8 TRUE_COLOR (cf=4) stores pixels at the display's native color depth —
    on an RGB565 display that is 2 bytes/pixel little-endian, the layout real
    device asset bins use. TRUE_COLOR_ALPHA (cf=5) is 4 bytes/pixel BGRA.
    Other v8 cfs raise ``LvglImageError``.
    """
    if len(d) < V8_HEADER_LEN:
        raise LvglImageError(f"v8 header truncated: {len(d)} bytes")
    v = struct.unpack_from("<I", d, 0)[0]
    cf = v & 0x1F
    w = (v >> 10) & 0x7FF
    h = (v >> 21) & 0x7FF
    body = d[V8_HEADER_LEN:]
    count = w * h
    if cf == V8_CF_TRUE_COLOR:
        need = count * 2
        if len(body) < need:
            raise LvglImageError(
                f"v8 cf=4 TRUE_COLOR body too short: got {len(body)} bytes, "
                f"need {need} (w={w} h={h} @2B/px RGB565)"
            )
        rgb = _rgb565_to_rgb(body[:need], count)
        return Image.frombytes("RGB", (w, h), rgb)
    if cf == V8_CF_TRUE_COLOR_ALPHA:
        need = count * 4
        if len(body) < need:
            raise LvglImageError(
                f"v8 cf=5 TRUE_COLOR_ALPHA body too short: got {len(body)} bytes, "
                f"need {need} (w={w} h={h} @4B/px BGRA)"
            )
        return Image.frombytes("RGBA", (w, h), body[:need], "raw", "BGRA")
    raise LvglImageError(
        f"v8 cf={cf} not handled (only cf=4 TRUE_COLOR RGB565 / "
        f"cf=5 TRUE_COLOR_ALPHA BGRA supported)"
    )


def decode_raw(data: bytes, w: int, h: int, pixel_fmt: str = "BGRA") -> Image.Image:
    """Decode headerless raw pixel data (vendor asset bins without headers).

    ``pixel_fmt`` is a PIL raw decoder name; real device assets are BGRA
    (4 B/px). Raises if the body does not exactly cover ``w * h`` pixels.
    """
    bpp = 4 if pixel_fmt in ("BGRA", "RGBA") else 2
    need = w * h * bpp
    if len(data) != need:
        raise LvglImageError(
            f"raw size mismatch: {len(data)} bytes, need {need} "
            f"({w}x{h} @ {bpp}B/px {pixel_fmt})"
        )
    mode = "RGBA" if bpp == 4 else "RGB"
    return Image.frombytes(mode, (w, h), data, "raw", pixel_fmt)


def detect_raw_size(data: bytes, lo: int = 16, hi: int = 1024) -> tuple[int, int]:
    """Guess (w, h) of a headerless BGRA buffer.

    Scans candidate widths (pixel count divisible) and picks the one where
    consecutive rows differ least — a wrong width misaligns rows into noise,
    the correct one keeps smooth image structure coherent.
    """
    if len(data) % 4:
        raise LvglImageError(f"raw size not BGRA-aligned: {len(data)} bytes")
    n = len(data) // 4
    if n < lo * lo:
        raise LvglImageError(f"raw buffer too small to guess: {n} px")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a soft dep
        raise LvglImageError("--raw auto requires numpy") from exc
    px = np.frombuffer(data, dtype=np.uint8).reshape(n, 4)[:, :3].astype(int)
    best: tuple[float, int, int] | None = None
    for w in range(lo, min(hi, n // lo) + 1):
        if n % w:
            continue
        h = n // w
        if h < lo:
            continue
        rows = px.reshape(h, w, 3)
        mad = np.abs(np.diff(rows, axis=0)).mean()
        if best is None or mad < best[0]:
            best = (mad, w, h)
    if best is None:
        raise LvglImageError(f"no candidate width in [{lo},{hi}] divides {n} px")
    return best[1], best[2]


def decode(data: bytes) -> Image.Image:
    """Decode LVGL ``.bin`` bytes (v8 or v9) into a PIL image.

    ``data`` must start at the file's first byte. v9 is selected by the 0x19
    magic byte; everything else is treated as a v8 4-byte bitfield header.
    """
    if not data:
        raise LvglImageError("empty input")
    if data[0] == MAGIC_V9:
        cf, w, h, stride, flags = _decode_v9_header(data)
        body = _maybe_decompress(data[12:], cf, flags)
        return _decode_v9_body(cf, w, h, stride, body)
    return _decode_v8(data)


def decode_file(path: str) -> Image.Image:
    """Convenience wrapper: read a ``.bin`` file and decode it."""
    with open(path, "rb") as f:
        return decode(f.read())

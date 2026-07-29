# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Junbo Zheng
"""Self-contained decoder tests — synthetic v9/v8 bins built in-test, no
external fixtures needed. Verifies round-trip correctness for NONE/RLE/LZ4
compression and the v8 true-color path."""

from __future__ import annotations

import struct

import lz4.block
import pytest
from PIL import Image

from lvglimg.decoder import (
    MAGIC_V9,
    LvglImageError,
    decode,
    decode_file,
)


def _make_rgba(w: int, h: int) -> Image.Image:
    """A deterministic RGBA image with varied pixels (exercises every channel)."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            px += bytes(
                ((x * 17) & 0xFF, (y * 31) & 0xFF, (x ^ y) & 0xFF, (x + y) & 0xFF)
            )
    return Image.frombytes("RGBA", (w, h), bytes(px))


def _v9_bin(cf_val: int, w: int, h: int, body: bytes, flags: int = 0) -> bytes:
    header = bytearray()
    header += bytes([MAGIC_V9, cf_val & 0x1F])
    header += flags.to_bytes(2, "little")
    header += w.to_bytes(2, "little")
    header += h.to_bytes(2, "little")
    stride = w * 4  # ARGB8888: 4 bytes/pixel, no padding
    header += stride.to_bytes(2, "little")
    header += (0).to_bytes(2, "little")  # reserved
    return bytes(header) + body


def _argb8888_body(img: Image.Image) -> bytes:
    """Pillow RGBA pixels are already R,G,B,A; LVGL ARGB8888 stores B,G,R,A."""
    r, g, b, a = img.split()
    return b"".join(
        bytes((B, G, R, A))
        for R, G, B, A in zip(
            r.tobytes(), g.tobytes(), b.tobytes(), a.tobytes(), strict=False
        )
    )


def _lz4_block(raw: bytes) -> bytes:
    comp = lz4.block.compress(raw, store_size=False)
    return (
        (2).to_bytes(4, "little")
        + len(comp).to_bytes(4, "little")
        + len(raw).to_bytes(4, "little")
        + comp
    )


def _rle_compress(raw: bytes, blksize: int = 4) -> bytes:
    """Minimal RLE encoder matching LVGL rle_compress (repeat vs literal)."""
    out = bytearray()
    i = 0
    n = len(raw)
    while i < n:
        blk = raw[i : i + blksize]
        run = 1
        while (
            i + run * blksize + blksize <= n
            and raw[i + run * blksize : i + (run + 1) * blksize] == blk
        ):
            run += 1
        if run >= 16:
            out.append(run & 0x7F)
            out += blk
            i += run * blksize
        else:
            # literal run until next repeat of >=16 or end
            j = i
            literal = 0
            while j < n and literal < 0x7F:
                cur = raw[j : j + blksize]
                ahead = 0
                while (
                    j + (ahead + 1) * blksize <= n
                    and raw[j + ahead * blksize : j + (ahead + 1) * blksize] == cur
                ):
                    ahead += 1
                if ahead >= 16:
                    break
                j += blksize
                literal += 1
            out.append(0x80 | literal)
            out += raw[i : i + literal * blksize]
            i += literal * blksize
    return bytes(out)


def _rle_block(raw: bytes) -> bytes:
    comp = _rle_compress(raw)
    return (
        (1).to_bytes(4, "little")
        + len(comp).to_bytes(4, "little")
        + len(raw).to_bytes(4, "little")
        + comp
    )


@pytest.mark.parametrize("w,h", [(1, 1), (5, 3), (16, 16)])
def test_argb8888_none(w: int, h: int):
    img = _make_rgba(w, h)
    body = _argb8888_body(img)
    bin_data = _v9_bin(0x10, w, h, body)  # ARGB8888
    out = decode(bin_data).convert("RGBA")
    assert out.size == (w, h)
    assert list(out.getdata()) == list(img.getdata())


def test_argb8888_stride_zero():
    """stride=0 is valid LVGL (default 1-byte-aligned stride). Must NOT
    collapse every row to row 0."""
    w, h = 4, 3
    img = _make_rgba(w, h)
    body = _argb8888_body(img)
    # build v9 header with stride=0 explicitly
    header = bytearray([MAGIC_V9, 0x10, 0, 0])
    header += w.to_bytes(2, "little")
    header += h.to_bytes(2, "little")
    header += (0).to_bytes(2, "little")  # stride = 0
    header += (0).to_bytes(2, "little")
    out = decode(bytes(header) + body).convert("RGBA")
    rows = [list(out.getdata())[y * w : (y + 1) * w] for y in range(h)]
    assert rows[0] != rows[1] != rows[2]
    assert list(out.getdata()) == list(img.getdata())


def test_argb8888_lz4():
    w, h = 12, 8
    img = _make_rgba(w, h)
    body = _argb8888_body(img)
    comp_block = _lz4_block(body)
    bin_data = _v9_bin(0x10, w, h, comp_block, flags=0x08)
    out = decode(bin_data).convert("RGBA")
    assert list(out.getdata()) == list(img.getdata())


def test_argb8888_rle():
    w, h = 10, 6
    img = _make_rgba(w, h)
    body = _argb8888_body(img)
    comp_block = _rle_block(body)
    bin_data = _v9_bin(0x10, w, h, comp_block, flags=0x08)
    out = decode(bin_data).convert("RGBA")
    assert list(out.getdata()) == list(img.getdata())


def test_v8_true_color_alpha():
    w, h = 7, 4
    img = _make_rgba(w, h)
    # v8 stores BGRA, identical byte order to Pillow RGBA for B,G,R,A layout
    rgba = img.tobytes()  # R,G,B,A
    bgra = b"".join(
        bytes((B, G, R, A))
        for R, G, B, A in zip(
            rgba[0::4], rgba[1::4], rgba[2::4], rgba[3::4], strict=False
        )
    )
    v = (5 & 0x1F) | ((w & 0x7FF) << 10) | ((h & 0x7FF) << 21)
    bin_data = struct.pack("<I", v) + bgra
    out = decode(bin_data).convert("RGBA")
    assert out.size == (w, h)
    assert list(out.getdata()) == list(img.getdata())


def test_v8_true_color_rgb565():
    """Real-device v8 cf=4 TRUE_COLOR images are RGB565 (2 B/px LE), not 4-byte BGRA.

    Real device asset bins (recovery screens etc.) use cf=4 at the display's
    native RGB565 depth. The pre-fix decoder assumed 4 B/px BGRA and raised
    "not enough image data"; this locks in the 2 B/px RGB565 path.
    """
    from lvglimg.decoder import _bit_extend

    w, h = 9, 5
    rgb = bytearray()
    for y in range(h):
        for x in range(w):
            rgb += bytes(((x * 17) & 0xFF, (y * 31) & 0xFF, (x ^ y) & 0xFF))
    img = Image.frombytes("RGB", (w, h), bytes(rgb))
    r, g, b = img.split()
    body = bytearray()
    fields = []
    for R, G, B in zip(r.tobytes(), g.tobytes(), b.tobytes(), strict=False):
        p = ((R >> 3) << 11) | ((G >> 2) << 5) | (B >> 3)
        fields.append(p)
        body += p.to_bytes(2, "little")
    v = (4 & 0x1F) | ((w & 0x7FF) << 10) | ((h & 0x7FF) << 21)
    bin_data = struct.pack("<I", v) + bytes(body)

    out = decode(bin_data)
    assert out.mode == "RGB"
    assert out.size == (w, h)

    expected = bytearray()
    for p in fields:
        expected += bytes(
            (
                _bit_extend((p >> 11) & 0x1F, 5),
                _bit_extend((p >> 5) & 0x3F, 6),
                _bit_extend(p & 0x1F, 5),
            )
        )
    assert out.tobytes() == bytes(expected)


def test_empty_input_raises():
    with pytest.raises(LvglImageError):
        decode(b"")


def test_unknown_cf_raises():
    # cf=0x00 (UNKNOWN) with a v9 header
    bin_data = _v9_bin(0x00, 2, 2, b"\x00" * 16)
    with pytest.raises(LvglImageError):
        decode(bin_data)


def test_decode_file(tmp_path):
    img = _make_rgba(4, 4)
    body = _argb8888_body(img)
    bin_data = _v9_bin(0x10, 4, 4, body)
    p = tmp_path / "t.bin"
    p.write_bytes(bin_data)
    out = decode_file(str(p)).convert("RGBA")
    assert list(out.getdata()) == list(img.getdata())


def _raw_bgra_bin(w: int, h: int) -> bytes:
    """Headerless BGRA buffer: each row a single color, color varies sharply
    per row — at the correct width rows are constant (MAD=0); any wrong width
    interleaves two colors per row (high MAD)."""
    out = bytearray()
    for y in range(h):
        for x in range(w):
            # Both x and y terms: a wrong width misaligns rows so the x term
            # shows up as row-to-row noise; the correct width keeps only the
            # small smooth y delta.
            out += bytes(
                (
                    (x * 7 + y * 37) & 0xFF,
                    (x * 11 + y * 53) & 0xFF,
                    (x * 13 + y * 71) & 0xFF,
                    0xFF,
                )
            )
    return bytes(out)


def test_decode_raw_roundtrip():
    from lvglimg.decoder import decode_raw

    data = _raw_bgra_bin(32, 20)
    img = decode_raw(data, 32, 20)
    assert img.size == (32, 20)
    assert img.getpixel((0, 5)) == (
        (5 * 71) & 0xFF,
        (5 * 53) & 0xFF,
        (5 * 37) & 0xFF,
        255,
    )


def test_decode_raw_size_mismatch_raises():
    from lvglimg.decoder import decode_raw

    with pytest.raises(LvglImageError, match="raw size mismatch"):
        decode_raw(_raw_bgra_bin(32, 20), 30, 20)


def test_detect_raw_size():
    from lvglimg.decoder import detect_raw_size

    assert detect_raw_size(_raw_bgra_bin(32, 20)) == (32, 20)


def test_detect_raw_size_not_bgra_aligned_raises():
    from lvglimg.decoder import detect_raw_size

    with pytest.raises(LvglImageError, match="not BGRA-aligned"):
        detect_raw_size(b"\x00" * 6)

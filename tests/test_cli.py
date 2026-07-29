# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Junbo Zheng
"""CLI smoke tests — version flag and a real bin→PNG conversion."""

from __future__ import annotations

import pytest

from lvglimg import __version__
from lvglimg.cli import main
from lvglimg.decoder import MAGIC_V9


def _argb8888_bin(w: int, h: int) -> bytes:
    body = bytes(i & 0xFF for i in range(w * h * 4))
    header = bytearray()
    header += bytes([MAGIC_V9, 0x10, 0x00, 0x00])  # cf=ARGB8888, flags=0
    header += w.to_bytes(2, "little")
    header += h.to_bytes(2, "little")
    header += (w * 4).to_bytes(2, "little")  # stride
    header += (0).to_bytes(2, "little")
    return bytes(header) + body


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_and_exits(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        main([flag])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_convert_file(tmp_path, capsys):
    bin_path = tmp_path / "a.bin"
    bin_path.write_bytes(_argb8888_bin(3, 2))
    out_path = tmp_path / "a.png"
    assert main([str(bin_path), str(out_path)]) == 0
    assert out_path.exists()
    from PIL import Image

    assert Image.open(out_path).size == (3, 2)


def test_default_output_path(tmp_path):
    bin_path = tmp_path / "a.bin"
    bin_path.write_bytes(_argb8888_bin(2, 2))
    assert main([str(bin_path)]) == 0
    assert (tmp_path / "a.png").exists()


def test_bad_file_returns_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x19\x00\x00\x00\x02\x00\x02\x00\x08\x00\x00\x00garbage")
    assert main([str(bad), str(tmp_path / "out.png")]) == 1


def test_batch_continues_on_bad_file(tmp_path, capsys):
    """A bad file must not crash the batch: print ERR, skip it, keep going,
    and exit non-zero. Regression for the uncaught-ValueError crash."""
    good = tmp_path / "good.bin"
    good.write_bytes(_argb8888_bin(2, 2))
    # valid-looking v9 header (ARGB8888) but truncated body -> PIL ValueError
    bad = tmp_path / "bad.bin"
    bad.write_bytes(bytes([0x19, 0x10, 0, 0, 2, 0, 2, 0, 8, 0, 0, 0]) + b"\x00\x00")
    out_dir = tmp_path / "preview"

    code = main([str(tmp_path), str(out_dir)])

    assert code == 1  # partial failure
    assert (out_dir / "good.png").exists()  # good file still converted
    err = capsys.readouterr().err
    assert "ERR" in err and "bad.bin" in err


def test_batch_no_bins_returns_2(tmp_path):
    assert main([str(tmp_path), str(tmp_path / "o")]) == 2


def test_batch_recursive(tmp_path):
    """Bins in subdirectories are found recursively and their relative
    directory structure is mirrored in the output dir."""
    sub = tmp_path / "icon"
    sub.mkdir()
    (tmp_path / "top.bin").write_bytes(_argb8888_bin(2, 2))
    (sub / "nested.bin").write_bytes(_argb8888_bin(3, 1))
    out_dir = tmp_path / "preview"

    assert main([str(tmp_path), str(out_dir)]) == 0

    assert (out_dir / "top.png").exists()
    assert (out_dir / "icon" / "nested.png").exists()  # subdir mirrored


def test_batch_copies_non_bin_images(tmp_path, capsys):
    """Non-.bin viewable images (e.g. .jpg) in the input dir are copied
    verbatim into the preview dir, mirroring subdirectory structure — no need
    to go back to the input dir to view them."""
    sub = tmp_path / "icon"
    sub.mkdir()
    (tmp_path / "a.bin").write_bytes(_argb8888_bin(2, 2))
    jpg_bytes = b"\xff\xd8\xff\xe0 fake jpg"
    (tmp_path / "b.jpg").write_bytes(jpg_bytes)
    (sub / "c.JPG").write_bytes(jpg_bytes)  # case-insensitive extension
    out_dir = tmp_path / "preview"

    assert main([str(tmp_path), str(out_dir)]) == 0

    assert (out_dir / "a.png").exists()
    assert (out_dir / "b.jpg").read_bytes() == jpg_bytes
    assert (out_dir / "icon" / "c.JPG").read_bytes() == jpg_bytes
    assert "2 copied" in capsys.readouterr().out


def test_default_output_dir_batch(tmp_path):
    """Default batch output is a sibling '<input>_preview' dir, not nested."""
    inp = tmp_path / "assets"
    inp.mkdir()
    (inp / "a.bin").write_bytes(_argb8888_bin(2, 2))

    assert main([str(inp)]) == 0

    assert (tmp_path / "assets_preview" / "a.png").exists()  # sibling of assets/
    assert not (inp / "assets_preview").exists()  # not nested inside input
    assert not (inp / "preview").exists()  # old name gone


def _raw_bgra_bin(w: int, h: int) -> bytes:
    out = bytearray()
    for y in range(h):
        for x in range(w):
            out += bytes(
                (
                    (x * 7 + y * 37) & 0xFF,
                    (x * 11 + y * 53) & 0xFF,
                    (x * 13 + y * 71) & 0xFF,
                    0xFF,
                )
            )
    return bytes(out)


def test_auto_fallback_on_raw_file(tmp_path, capsys):
    # Headerless BGRA: no v9 magic, v8 parse fails → auto raw fallback.
    bin_path = tmp_path / "boot.bin"
    bin_path.write_bytes(_raw_bgra_bin(32, 20))
    out_path = tmp_path / "boot.png"
    assert main([str(bin_path), str(out_path)]) == 0
    assert "raw auto-detected 32x20" in capsys.readouterr().out
    from PIL import Image

    assert Image.open(out_path).size == (32, 20)


def test_raw_explicit_size(tmp_path, capsys):
    bin_path = tmp_path / "boot.bin"
    bin_path.write_bytes(_raw_bgra_bin(32, 20))
    out_path = tmp_path / "boot.png"
    assert main(["--raw", "32x20", str(bin_path), str(out_path)]) == 0
    assert "raw auto-detected" not in capsys.readouterr().out
    from PIL import Image

    assert Image.open(out_path).size == (32, 20)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Junbo Zheng
"""Command-line entry point for :mod:`lvglimg`."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version

from .decoder import LvglImageError, decode, decode_raw, detect_raw_size

# Viewable raster images sometimes shipped alongside the .bin assets. They
# need no decoding — batch mode copies them verbatim into the preview dir so
# it holds the full asset set.
COPY_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def _version() -> str:
    try:
        return version("lvglimg")
    except PackageNotFoundError:
        from . import __version__

        return __version__


def _bin_to_png(bin_path: str, out_path: str, raw: str | None = None) -> str:
    with open(bin_path, "rb") as f:
        data = f.read()
    if raw:
        if raw == "auto":
            w, h = detect_raw_size(data)
        else:
            try:
                w_s, h_s = raw.lower().split("x")
                w, h = int(w_s), int(h_s)
            except ValueError as exc:
                raise SystemExit(
                    f"invalid --raw '{raw}' (expected WxH or auto)"
                ) from exc
        img = decode_raw(data, w, h)
    else:
        try:
            img = decode(data)
        except LvglImageError:
            # No magic + unparseable v8 header → likely a headerless raw
            # buffer. Retry with a guessed size.
            w, h = detect_raw_size(data)
            img = decode_raw(data, w, h)
            print(f"    [raw auto-detected {w}x{h} BGRA: {bin_path}]")
    img.save(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lvglimg",
        description=f"lvglimg {_version()} — decode LVGL .bin images (v8+v9) to PNG.",
    )
    p.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"lvglimg {_version()}",
        help="Show the version and exit",
    )
    p.add_argument(
        "--raw",
        metavar="WxH|auto",
        default=None,
        help=(
            "Treat input as headerless raw BGRA pixels, not LVGL .bin. "
            "'auto' guesses the size via row-coherence scan."
        ),
    )
    p.add_argument(
        "input",
        help="A .bin file, or a directory of .bin files (batch mode).",
    )
    p.add_argument(
        "output",
        nargs="?",
        help=(
            "Output PNG path (file mode) or output directory (batch mode). "
            "Defaults to a sibling .png / a sibling '<input>_preview' dir."
        ),
    )
    args = p.parse_args(argv)

    if os.path.isdir(args.input):
        import glob

        # Recursive: find .bin at top level AND in any subdirectory.
        bins = sorted(
            glob.glob(os.path.join(args.input, "**", "*.bin"), recursive=True)
        )
        norm = os.path.normpath(args.input)
        in_parent = os.path.dirname(norm)
        name = os.path.basename(norm)
        out_dir = args.output or os.path.join(in_parent or ".", f"{name}_preview")
        os.makedirs(out_dir, exist_ok=True)
        # Copy already-viewable images (e.g. .jpg) verbatim, mirroring the
        # subdirectory structure, BEFORE decoding so a decoded .png wins a
        # same-stem name collision. Never descend into out_dir itself (it may
        # sit inside the input dir on a re-run).
        out_real = os.path.realpath(out_dir)
        copied = 0
        for root, subdirs, files in os.walk(args.input):
            subdirs[:] = [
                d
                for d in subdirs
                if os.path.realpath(os.path.join(root, d)) != out_real
            ]
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in COPY_EXTS:
                    continue
                src = os.path.join(root, fn)
                dst = os.path.join(out_dir, os.path.relpath(src, args.input))
                parent = os.path.dirname(dst)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
                print(f"CPY {src} -> {dst}")
        failures = 0
        failed = []  # (bin_path, exc) for the end-of-run summary
        for b in bins:
            # Preserve subdirectory structure in the output dir
            # (e.g. input/icon/a.bin -> out/icon/a.png), avoiding name collisions.
            rel = os.path.relpath(b, args.input)
            out = os.path.join(out_dir, os.path.splitext(rel)[0] + ".png")
            parent = os.path.dirname(out)
            if parent:
                os.makedirs(parent, exist_ok=True)
            try:
                _bin_to_png(b, out, raw=args.raw)
                print(f"OK  {b} -> {out}")
            except Exception as exc:  # noqa: BLE001 - CLI must not crash on one bad file
                failures += 1
                failed.append((b, exc))
                print(f"ERR {b}: {exc}", file=sys.stderr)
        if not bins:
            return 2
        dirs = len({os.path.dirname(os.path.relpath(b, args.input)) for b in bins})
        ok = len(bins) - failures
        print(
            f"\nDone: {len(bins)} files in {dirs} dirs, {ok} ok, "
            f"{failures} failed, {copied} copied",
            file=sys.stderr if failures else sys.stdout,
        )
        if failed:
            # Group failed files by their directory for a scannable summary.
            by_dir: dict[str, list[str]] = {}
            for b, _ in failed:
                by_dir.setdefault(os.path.dirname(b) or ".", []).append(b)
            print(f"\nFailed ({failures}, in {len(by_dir)} dirs):", file=sys.stderr)
            for d in sorted(by_dir):
                print(f"  {d}/ ({len(by_dir[d])})", file=sys.stderr)
                for b in sorted(by_dir[d]):
                    print(f"    {os.path.basename(b)}", file=sys.stderr)
        return 1 if failures else 0

    out_path = args.output or os.path.splitext(args.input)[0] + ".png"
    try:
        print(_bin_to_png(args.input, out_path, raw=args.raw))
    except Exception as exc:  # noqa: BLE001 - report any decode failure, not just known ones
        print(f"ERR {args.input}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

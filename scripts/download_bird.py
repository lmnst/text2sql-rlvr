"""Download and unpack BIRD into the layout the rest of the project expects.

    python scripts/download_bird.py --split mini_dev

Start with mini_dev (500 questions, 11 databases). Only fetch dev when you are
ready to report a number, and train when you are ready to do SFT -- both are
much larger, and dev in particular should be touched as rarely as possible
(see "数据划分纪律" in AGENTS.md).

BIRD ships the databases as a zip *inside* the outer zip, so unpacking is two
passes. Downloads resume: rerun after a dropped connection and it continues.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from text2sql_rlvr.data import discover_split

URLS = {
    "mini_dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip",
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
}

_MB = 1024 * 1024


def human(n: int) -> str:
    return f"{n / _MB:.1f} MB" if n < 4096 * _MB else f"{n / (1024 * _MB):.2f} GB"


def download(url: str, target: Path) -> Path:
    """Fetch ``url`` to ``target``, resuming a partial file if one is there."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))

    if target.is_file() and (not total or target.stat().st_size == total):
        print(f"  already downloaded: {target.name} ({human(target.stat().st_size)})")
        return target

    have = partial.stat().st_size if partial.is_file() else 0
    if have and total and have < total:
        print(f"  resuming at {human(have)} of {human(total)}")
    else:
        have = 0
        partial.unlink(missing_ok=True)

    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")

    print(f"  downloading {url}")
    print(f"  size {human(total) if total else 'unknown'} -> {target}")
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if have and response.status == 206 else "wb"
        if mode == "wb":
            have = 0
        with partial.open(mode) as handle:
            done = have
            while chunk := response.read(1 * _MB):
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {human(done)} / {human(total)}  ({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {human(done)}", end="", flush=True)
    print()

    partial.replace(target)
    return target


def extract_all(archive: Path, destination: Path, passes: int = 2) -> None:
    """Unzip ``archive``, then unzip any zips it contained. BIRD nests them."""
    destination.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {destination}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)

    for _ in range(passes - 1):
        inner = [
            p
            for p in destination.rglob("*.zip")
            if p != archive and "__MACOSX" not in p.parts
        ]
        if not inner:
            break
        for nested in inner:
            print(f"  extracting nested {nested.relative_to(destination)}")
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(nested.parent)
            nested.unlink()

    junk = destination / "__MACOSX"
    if junk.is_dir():
        shutil.rmtree(junk, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=sorted(URLS), default="mini_dev")
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    parser.add_argument("--keep-zip", action="store_true", help="do not delete the archive")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    url = URLS[args.split]

    print(f"split {args.split}")
    archive = download(url, args.root / Path(url).name)
    extract_all(archive, args.root)
    if not args.keep_zip:
        archive.unlink(missing_ok=True)

    try:
        split = discover_split(args.root, args.split)
    except FileNotFoundError as exc:
        print(f"\nunpacked, but the layout was not recognised:\n  {exc}")
        print("List what landed in the root and check docs/data.md.")
        return 1

    examples = split.load()
    missing = split.missing_databases(examples)
    print("\nready")
    print(f"  questions  {split.questions_path}")
    print(f"  databases  {split.databases_dir}")
    print(f"  examples   {len(examples)}")
    if missing:
        print(f"  MISSING DATABASES ({len(missing)}): {', '.join(missing)}")
        return 1

    print("\nnext:")
    print(f"  python scripts/prepare_bird.py --root {args.root} "
          f"--split {args.split} --check-gold")
    return 0


if __name__ == "__main__":
    sys.exit(main())

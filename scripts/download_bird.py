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


def verify_zip(path: Path) -> str | None:
    """Return ``None`` if the archive is intact, else why it is not.

    Worth the CRC pass over the whole file. A resumed download whose seam is
    misaligned still has a valid central directory at the end, so opening the
    zip succeeds and only extraction fails -- several hundred megabytes later,
    with an error that points at the zip module instead of at the download.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            broken = zf.testzip()
    except zipfile.BadZipFile as exc:
        return f"not a valid zip ({exc})"
    except OSError as exc:
        return f"unreadable ({exc})"
    return f"corrupt member {broken}" if broken else None


def download(url: str, target: Path, *, allow_resume: bool = True) -> Path:
    """Fetch ``url`` to ``target``, optionally resuming a partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))

    if target.is_file() and (not total or target.stat().st_size == total):
        print(f"  already downloaded: {target.name} ({human(target.stat().st_size)})")
        return target

    have = partial.stat().st_size if partial.is_file() else 0
    if allow_resume and have and total and have < total:
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
        resuming = have > 0 and response.status == 206
        if resuming:
            # Trust the server's answer, not our request: a proxy or CDN may
            # serve a different range than the one we asked for, and appending
            # it would corrupt the seam silently.
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {have}-"):
                print(f"  server returned {content_range!r}, restarting from zero")
                resuming = False
        if not resuming:
            have = 0
        with partial.open("ab" if resuming else "wb") as handle:
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


def fetch_archive(url: str, target: Path, *, force: bool = False, attempts: int = 2) -> Path:
    """Download and verify, discarding and refetching once if verification fails."""
    partial = target.with_suffix(target.suffix + ".part")
    if force:
        target.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    for attempt in range(1, attempts + 1):
        download(url, target, allow_resume=attempt == 1 and not force)
        print("  verifying archive...")
        problem = verify_zip(target)
        if problem is None:
            print("  archive ok")
            return target

        print(f"  ARCHIVE FAILED VERIFICATION: {problem}")
        target.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        if attempt < attempts:
            print("  discarded; downloading again from scratch")

    raise RuntimeError(
        f"could not obtain an intact archive from {url} after {attempts} attempts. "
        "The link to the Beijing OSS bucket may be too unreliable from here -- try "
        "the HuggingFace mirror with --url, see docs/data.md."
    )


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
    parser.add_argument("--url", help="download from a mirror instead of the official bucket")
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard any existing or partial archive and download from scratch",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    url = args.url or URLS[args.split]

    print(f"split {args.split}")
    try:
        archive = fetch_archive(url, args.root / Path(url).name, force=args.force)
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1
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

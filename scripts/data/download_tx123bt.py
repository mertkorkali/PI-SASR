#!/usr/bin/env python
"""Download the TX-123BT five-year profiles and extract the files the paper uses.

Source: J. Lu and X. Li, "Texas Synthetic Power System Test Case (TX-123BT).zip,"
figshare, dataset, version 6, 2023, doi:10.6084/m9.figshare.22144616.v6, licensed
CC BY 4.0.
The archive (544 MB) is checked against its published MD5 checksum, and only
``Generator_data.xlsx``, ``Readme.txt`` and the ``Load_5y``, ``Wind_5y`` and
``Solar_5y`` folders (about 80 MB) are extracted to ``<dest>`` (default
``data/raw/TX-123BT``). These are the inputs of ``scripts/data/prepare_tx123bt.py``.

Example::

    python scripts/data/download_tx123bt.py
    python scripts/data/download_tx123bt.py --zip "/path/to/TX-123BT 5year profiles.zip"
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from pisasr import paths

URL = "https://ndownloader.figshare.com/files/44942761"
MD5 = "60db05d9f7f7699380c8bd023b38ba43"
ARCHIVE_ROOT = "Data_public_5year/"
MEMBERS = ("Generator_data.xlsx", "Readme.txt", "Load_5y/", "Wind_5y/", "Solar_5y/")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download (or reuse) the TX-123BT figshare archive, check its MD5 "
                    "checksum and extract the generator data and five-year profiles.",
    )
    parser.add_argument("--dest", type=Path, default=paths.RAW_DIR / "TX-123BT",
                        help="extraction folder (default: data/raw/TX-123BT in the repository, "
                             "or $PISASR_RAW_DIR/TX-123BT)")
    parser.add_argument("--zip", type=Path, default=None,
                        help="use an already downloaded archive instead of downloading")
    parser.add_argument("--keep-zip", action="store_true",
                        help="keep the downloaded archive next to the extraction folder")
    parser.add_argument("--force", action="store_true",
                        help="extract again even if the destination already holds the files")
    return parser.parse_args(argv)


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    print(f"Downloading {url}\n  to {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} of {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(target)


def extract(archive: Path, dest: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name.startswith(ARCHIVE_ROOT) or info.is_dir():
                continue
            rel = name[len(ARCHIVE_ROOT):]
            if not rel.startswith(MEMBERS) or Path(rel).name == ".DS_Store":
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def main(argv=None) -> int:
    args = parse_args(argv)
    dest = args.dest
    if (dest / "Generator_data.xlsx").exists() and (dest / "Load_5y").exists() and not args.force:
        print(f"{dest} already holds the TX-123BT files; nothing to do (use --force).")
        return 0

    if args.zip is not None:
        archive = args.zip.expanduser()
        if not archive.is_file():
            print(f"Error: archive not found: {archive}", file=sys.stderr)
            return 1
        downloaded = False
    else:
        archive = dest.parent / "TX-123BT_5year_profiles.zip"
        if not archive.exists():
            download(URL, archive)
        downloaded = True

    print(f"Checking MD5 of {archive} ...")
    actual = md5_file(archive)
    if actual != MD5:
        print(f"MD5 mismatch: expected {MD5}, found {actual}")
        return 1
    print(f"MD5 OK ({MD5})")

    print(f"Extracting {', '.join(MEMBERS)} to {dest} ...")
    n = extract(archive, dest)
    print(f"Extracted {n} files.")

    if downloaded and not args.keep_zip:
        archive.unlink()
        print(f"Removed {archive}")
    print("Next step: python scripts/data/prepare_tx123bt.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())

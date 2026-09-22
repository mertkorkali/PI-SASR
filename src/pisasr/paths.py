"""Repository folders and data checksums.

All locations are derived from the location of this file, so the repository can
be cloned anywhere. Two environment variables override the defaults:

``PISASR_OUTPUT_DIR``
    destination of every rerun (default ``<repository>/outputs``);
``PISASR_RAW_DIR``
    location of the raw third-party data used to regenerate the processed
    inputs (default ``<repository>/data/raw``).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Root of the repository (the folder that contains ``src/``)."""

DATA_DIR = REPO_ROOT / "data"
"""Processed input data included in the repository."""

REFERENCE_DIR = REPO_ROOT / "results"
"""Reference results of the paper (read-only for reruns)."""

OUTPUT_DIR = Path(os.environ.get("PISASR_OUTPUT_DIR", str(REPO_ROOT / "outputs"))).expanduser().resolve()
"""Default destination of every rerun (a relative ``PISASR_OUTPUT_DIR`` is taken
relative to the working directory and made absolute here)."""

RAW_DIR = Path(os.environ.get("PISASR_RAW_DIR", str(DATA_DIR / "raw"))).expanduser().resolve()
"""Raw third-party data (not included; see ``data/README.md``)."""

CHECKSUM_FILE = DATA_DIR / "SHA256SUMS"
"""SHA-256 checksums of the processed data files, relative to ``data/``."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(checksum_file: str | Path = CHECKSUM_FILE) -> dict[str, str]:
    """Read a ``sha256sum``-style file into ``{relative path: digest}``."""
    sums: dict[str, str] = {}
    for line in Path(checksum_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*").strip()] = digest
    return sums


def verify_checksums(
    target_dir: str | Path = DATA_DIR,
    prefix: str | None = None,
    checksum_file: str | Path = CHECKSUM_FILE,
    verbose: bool = True,
) -> bool:
    """Compare files against ``data/SHA256SUMS``.

    Parameters
    ----------
    target_dir
        Folder that holds the files to check. With ``prefix=None`` it plays the
        role of ``data/``; with ``prefix="tx123bt"`` it plays the role of
        ``data/tx123bt/`` (useful for a regenerated copy of one system).
    prefix
        Restrict the check to entries below this sub-folder of ``data/``.
    checksum_file
        The checksum file to compare against.
    verbose
        Print one line for each file.

    Returns
    -------
    bool
        ``True`` when every listed file exists and matches its checksum.
    """
    target_dir = Path(target_dir)
    sums = read_checksums(checksum_file)
    ok_all = True
    n_checked = 0
    for name, expected in sorted(sums.items()):
        if prefix is not None:
            head = prefix.rstrip("/") + "/"
            if not name.startswith(head):
                continue
            path = target_dir / name[len(head):]
        else:
            path = target_dir / name
        n_checked += 1
        if not path.exists():
            ok_all = False
            if verbose:
                print(f"MISSING   {name}  ({path})")
            continue
        actual = sha256_file(path)
        ok = actual == expected
        ok_all &= ok
        if verbose:
            print(f"{'OK' if ok else 'MISMATCH':9s} {name}")
    if n_checked == 0:
        if verbose:
            print(f"No entries of {checksum_file} lie below {prefix!r}.")
        return False
    if verbose:
        print(f"{n_checked} file(s) checked: {'all match' if ok_all else 'differences found'}.")
    return ok_all


def absolute_path(path: str | Path) -> Path:
    """Return ``path`` as an absolute path (a relative path is taken relative to
    the working directory), so that later joins cannot apply it twice."""
    return Path(path).expanduser().resolve()


PACKAGE_DIR = Path(__file__).resolve().parent
"""Folder of the installed package (``<repository>/src/pisasr`` in an editable install)."""


class LayoutError(FileNotFoundError):
    """The repository folder ``data/`` or ``results/`` cannot be found."""


def check_layout(folders=None) -> None:
    """Raise :class:`LayoutError` when repository folders cannot be found.

    The package locates ``data/`` and ``results/`` relative to its source folder,
    so it must be installed in editable mode (``pip install -e .``) from a clone
    of the repository.

    Parameters
    ----------
    folders
        Folders that must exist (default: ``data/`` and ``results/``).
    """
    folders = [DATA_DIR, REFERENCE_DIR] if folders is None else [Path(f) for f in folders]
    missing = [f for f in folders if not f.is_dir()]
    if missing:
        names = " and ".join(f"{f.name}/" for f in missing)
        raise LayoutError(
            f"Cannot find {names} in {REPO_ROOT}, two levels above the package folder {PACKAGE_DIR}. "
            "Install the package in editable mode from a clone of the repository "
            "(python -m pip install -e .)."
        )


def run_script(main) -> int:
    """Run the ``main`` function of a script and return its exit status.

    A :class:`LayoutError` is reported as a one-line message on standard error,
    with exit status 1, instead of a traceback.
    """
    try:
        return main()
    except LayoutError as exc:
        sys.stdout.flush()
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def display_path(path: str | Path) -> str:
    """Return ``path`` relative to the repository root when it lies inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

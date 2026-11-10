#!/usr/bin/env python3
"""Remove local account metadata from explicitly supplied source distributions.

Usage: python scripts/normalize_sdist.py ARCHIVE.tar.gz [ARCHIVE2.tar.gz ...]

Each gzip-compressed tar archive is rewritten atomically in its own directory.
Member payloads, order, names, links, modes, modification times and non-owner PAX
metadata are retained. Ownership becomes root (UID/GID 0), owner PAX overrides are
removed, and the gzip filename field is omitted. No member paths are extracted to
the filesystem. A failed rewrite leaves that archive's original bytes untouched.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile


OWNER_PAX_KEYS = frozenset({"uid", "gid", "uname", "gname",
                            "SCHILY.uid", "SCHILY.gid", "SCHILY.uname", "SCHILY.gname"})


def _without_owner(headers: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key not in OWNER_PAX_KEYS}


def normalize_sdist(archive: str | Path) -> None:
    """Replace one regular gzip tar only after its normalized rewrite succeeds.

    Archive-level permissions and timestamps are retained as well. Sparse GNU
    members are rejected rather than silently changing their representation;
    ordinary Python source distributions do not require sparse members.
    """
    archive = Path(archive).expanduser()
    attributes = archive.lstat()
    if not stat.S_ISREG(attributes.st_mode):
        raise ValueError(f"Archive must be a regular file, not a link or directory: {archive}")
    temporary: Path | None = None
    try:
        with archive.open("rb") as source_file, gzip.GzipFile(fileobj=source_file, mode="rb") as source_gzip:
            with tarfile.open(fileobj=source_gzip, mode="r:") as source_tar:
                with tempfile.NamedTemporaryFile(mode="w+b", dir=archive.parent,
                                                 prefix=f".{archive.name}.", suffix=".tmp",
                                                 delete=False) as destination_file:
                    temporary = Path(destination_file.name)
                    with gzip.GzipFile(filename="", fileobj=destination_file, mode="wb",
                                       mtime=source_gzip.mtime, compresslevel=9) as destination_gzip:
                        with tarfile.open(fileobj=destination_gzip, mode="w", format=tarfile.PAX_FORMAT,
                                          pax_headers=_without_owner(source_tar.pax_headers)) as destination_tar:
                            for member in source_tar:
                                if member.issparse():
                                    raise ValueError("Sparse tar members are not supported for source distributions")
                                if not member.isfile() and member.size:
                                    raise ValueError("Non-file tar members with payloads are not supported")
                                normalized = copy.copy(member)
                                normalized.uid = normalized.gid = 0
                                normalized.uname = normalized.gname = "root"
                                normalized.pax_headers = _without_owner(member.pax_headers)
                                # extractfile returns a payload stream; it never writes a member path.
                                if member.isfile():
                                    with source_tar.extractfile(member) as payload:
                                        destination_tar.addfile(normalized, payload)
                                else:
                                    destination_tar.addfile(normalized)
                        # Force gzip CRC/trailer verification, including unread tar padding.
                        while source_gzip.read(1024 * 1024):
                            pass
                    destination_file.flush()
                    os.fsync(destination_file.fileno())
        os.chmod(temporary, stat.S_IMODE(attributes.st_mode))
        os.utime(temporary, ns=(attributes.st_atime_ns, attributes.st_mtime_ns))
        os.replace(temporary, archive)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Normalize each named archive; return nonzero on the first failed archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path,
                        metavar="ARCHIVE.tar.gz", help="Built gzip tar archive to normalize in place")
    arguments = parser.parse_args(argv)
    for archive in arguments.archives:
        try:
            normalize_sdist(archive)
        except (OSError, ValueError, EOFError, tarfile.TarError) as error:
            print(f"Could not normalize {archive}: {error}", file=sys.stderr)
            return 1
        print(f"Normalized {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

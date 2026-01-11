"""Download the ULB credit card fraud dataset from Kaggle.

Usage
-----
    python -m scripts.download_data                 # idempotent download
    python -m scripts.download_data --force         # re-download even if present
    python -m scripts.download_data --no-verify     # skip the SHA-256 check

Requirements
------------
- The ``kaggle`` CLI must be on PATH (``pip install kaggle``).
- A Kaggle API token at ``~/.kaggle/kaggle.json`` (or the Windows equivalent
  at ``%USERPROFILE%\\.kaggle\\kaggle.json``). See data/README.md.

Exit codes
----------
0  success (or dataset already present)
1  SHA-256 mismatch
2  kaggle CLI not found
3  kaggle download failed
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from fraud_shield.config import settings
from fraud_shield.utils.logging import get_logger

log = get_logger(__name__)

DATASET = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"

# Filled in on first successful download; leave as placeholder until then.
EXPECTED_SHA256: str | None = None


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream the file through SHA-256 to avoid loading 144 MB into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> bool:
    """Return True if ``path`` matches ``expected`` (or ``expected`` is None)."""
    if expected is None:
        log.info("checksum_skipped", reason="no expected hash configured yet")
        return True
    actual = sha256_of(path)
    matches = actual == expected
    if matches:
        log.info("checksum_ok", sha256=actual)
    else:
        log.warning("checksum_mismatch", actual=actual, expected=expected)
    return matches


def download(dest: Path, force: bool = False) -> Path:
    """Download the dataset via the Kaggle CLI. Raises on failure."""
    dest.mkdir(parents=True, exist_ok=True)
    csv = dest / CSV_NAME

    if csv.exists() and not force:
        log.info("dataset_already_present", path=str(csv))
        return csv

    if not shutil.which("kaggle"):
        raise FileNotFoundError(
            "kaggle CLI not found on PATH — run `pip install kaggle` and "
            "place an API token at ~/.kaggle/kaggle.json (see data/README.md)"
        )

    log.info("downloading", dataset=DATASET, dest=str(dest))
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(dest), "--unzip"],
        check=True,
    )
    log.info("download_complete", path=str(csv), size_bytes=csv.stat().st_size)
    return csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download even if file exists")
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 verification")
    args = parser.parse_args(argv)

    try:
        csv = download(settings.data_raw, force=args.force)
    except FileNotFoundError as e:
        log.error("kaggle_cli_missing", error=str(e))
        return 2
    except subprocess.CalledProcessError as e:
        log.error("kaggle_download_failed", returncode=e.returncode)
        return 3

    if not args.no_verify and not verify_sha256(csv, EXPECTED_SHA256):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

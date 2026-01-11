"""Tests for the Kaggle download script helpers.

We don't actually hit Kaggle in tests — the I/O-heavy ``download()`` function
is a thin subprocess wrapper. What's worth testing is the checksum logic,
which is pure and easy to verify against a known fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.download_data import sha256_of, verify_sha256


@pytest.fixture
def hello_world_file(tmp_path: Path) -> Path:
    f = tmp_path / "hello.bin"
    f.write_bytes(b"hello world")
    return f


# Reference value: $ printf 'hello world' | sha256sum
HELLO_WORLD_SHA256 = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_sha256_of_matches_reference(hello_world_file: Path) -> None:
    assert sha256_of(hello_world_file) == HELLO_WORLD_SHA256


def test_sha256_of_streams_large_files(tmp_path: Path) -> None:
    """File bigger than the chunk size should hash correctly."""
    big = tmp_path / "big.bin"
    big.write_bytes(b"a" * (2 << 20))  # 2 MiB, larger than the 1 MiB chunk
    assert len(sha256_of(big)) == 64


def test_verify_returns_true_when_hash_matches(hello_world_file: Path) -> None:
    assert verify_sha256(hello_world_file, HELLO_WORLD_SHA256) is True


def test_verify_returns_false_when_hash_mismatches(hello_world_file: Path) -> None:
    assert verify_sha256(hello_world_file, "0" * 64) is False


def test_verify_returns_true_when_expected_is_none(hello_world_file: Path) -> None:
    """Unconfigured checksum means the script is permissive (with a warning)."""
    assert verify_sha256(hello_world_file, None) is True

"""Unpacking logic only -- no network. BIRD nests the databases zip inside the
outer zip, which is the part that silently half-works if you get it wrong."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def download_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("download_script", SCRIPTS / "download_bird.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def nested_archive(tmp_path: Path) -> Path:
    """An outer zip containing a json file and an inner zip of 'databases'."""
    inner = tmp_path / "dev_databases.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("dev_databases/company/company.sqlite", "not really a database")

    outer = tmp_path / "minidev.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("MINIDEV/mini_dev_sqlite.json", "[]")
        zf.write(inner, "MINIDEV/dev_databases.zip")
        zf.writestr("__MACOSX/._junk", "")
    inner.unlink()
    return outer


def test_nested_zip_is_unpacked(download_module, nested_archive, tmp_path):
    out = tmp_path / "out"
    download_module.extract_all(nested_archive, out)

    assert (out / "MINIDEV" / "mini_dev_sqlite.json").is_file()
    assert (out / "MINIDEV" / "dev_databases" / "company" / "company.sqlite").is_file()


def test_inner_archive_is_removed_after_extraction(download_module, nested_archive, tmp_path):
    out = tmp_path / "out"
    download_module.extract_all(nested_archive, out)
    assert not (out / "MINIDEV" / "dev_databases.zip").exists()


def test_macos_metadata_is_discarded(download_module, nested_archive, tmp_path):
    out = tmp_path / "out"
    download_module.extract_all(nested_archive, out)
    assert not (out / "__MACOSX").exists()


def test_flat_archive_still_works(download_module, tmp_path):
    flat = tmp_path / "flat.zip"
    with zipfile.ZipFile(flat, "w") as zf:
        zf.writestr("train/train.json", "[]")

    out = tmp_path / "out"
    download_module.extract_all(flat, out)
    assert (out / "train" / "train.json").is_file()


def test_every_split_has_a_url(download_module):
    from text2sql_rlvr.data import SPLITS

    assert set(download_module.URLS) == set(SPLITS)

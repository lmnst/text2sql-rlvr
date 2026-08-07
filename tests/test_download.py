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


class TestVerification:
    """A misaligned resume leaves the central directory intact, so the zip opens
    fine and only blows up mid-extraction. Verification has to catch that."""

    def _good_zip(self, tmp_path: Path) -> Path:
        # Stored, not deflated, and large enough that the payload dwarfs the
        # central directory -- otherwise "damage the middle" lands in the
        # directory and produces a different, easier failure.
        path = tmp_path / "good.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("a.bin", bytes(range(256)) * 400)
            zf.writestr("b/c.bin", bytes(range(256)) * 400)
        return path

    def test_intact_archive_passes(self, download_module, tmp_path):
        assert download_module.verify_zip(self._good_zip(tmp_path)) is None

    def test_corrupt_payload_is_caught(self, download_module, tmp_path):
        path = self._good_zip(tmp_path)
        data = bytearray(path.read_bytes())
        # Damage the first member's payload, the way a bad resume seam does,
        # leaving the central directory at the end of the file untouched.
        data[5000:5064] = b"\x00" * 64
        path.write_bytes(data)

        with zipfile.ZipFile(path) as zf:  # still opens: this is the trap
            assert zf.namelist() == ["a.bin", "b/c.bin"]
        assert download_module.verify_zip(path) is not None

    def test_non_zip_is_caught(self, download_module, tmp_path):
        path = tmp_path / "nope.zip"
        path.write_bytes(b"this is not a zip file")
        assert "not a valid zip" in download_module.verify_zip(path)

    def test_missing_file_is_caught(self, download_module, tmp_path):
        assert download_module.verify_zip(tmp_path / "absent.zip") is not None


class TestFetchRetry:
    def _fake_download(self, download_module, monkeypatch, payloads):
        """Replace the network call with a sequence of file bodies to write."""
        calls = []

        def fake(url, target, *, allow_resume=True):
            calls.append({"url": url, "allow_resume": allow_resume})
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payloads[len(calls) - 1])
            return target

        monkeypatch.setattr(download_module, "download", fake)
        return calls

    def _zip_bytes(self, tmp_path: Path) -> bytes:
        path = tmp_path / "src.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("a.json", "[]")
        return path.read_bytes()

    def test_a_corrupt_archive_is_refetched_without_resume(
        self, download_module, monkeypatch, tmp_path
    ):
        good = self._zip_bytes(tmp_path)
        calls = self._fake_download(download_module, monkeypatch, [b"garbage", good])

        target = tmp_path / "out.zip"
        assert download_module.fetch_archive("http://x/a.zip", target) == target
        assert [c["allow_resume"] for c in calls] == [True, False]

    def test_persistent_corruption_raises_with_advice(
        self, download_module, monkeypatch, tmp_path
    ):
        self._fake_download(download_module, monkeypatch, [b"garbage", b"still garbage"])

        with pytest.raises(RuntimeError, match="after 2 attempts"):
            download_module.fetch_archive("http://x/a.zip", tmp_path / "out.zip")

    def test_force_skips_resume_on_the_first_attempt(
        self, download_module, monkeypatch, tmp_path
    ):
        calls = self._fake_download(download_module, monkeypatch, [self._zip_bytes(tmp_path)])

        target = tmp_path / "out.zip"
        target.write_bytes(b"stale")
        download_module.fetch_archive("http://x/a.zip", target, force=True)
        assert calls[0]["allow_resume"] is False

    def test_failed_archive_is_not_left_behind(self, download_module, monkeypatch, tmp_path):
        self._fake_download(download_module, monkeypatch, [b"garbage", b"garbage"])
        target = tmp_path / "out.zip"

        with pytest.raises(RuntimeError):
            download_module.fetch_archive("http://x/a.zip", target)
        assert not target.exists()

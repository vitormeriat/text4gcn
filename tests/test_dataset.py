"""Tests for the datasets module."""

import hashlib

import pytest

from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger
from text4gcn.datasets import data


def test_gets_data_list_len():
    """Test that the data list contains the expected datasets."""
    assert data.list() == ['R8', 'R52', 'ohsumed', 'pubmed']


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content


def test_download_fetches_txt_and_meta_from_the_release(tmp_path, monkeypatch):
    files = {"toy.txt": b"a doc\n", "toy.meta": b"0\ttrain\tx\n"}
    monkeypatch.setattr(data, "BENCHMARKS", {
        "toy": tuple(hashlib.sha256(files[f]).hexdigest() for f in ("toy.txt", "toy.meta"))})
    requested = []

    def fake_get(url, **kwargs):
        requested.append(url)
        return _FakeResponse(files[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(data.requests, "get", fake_get)
    txt, meta = data.download("toy", tmp_path)

    assert requested == [f"{data.BASE_URL}/toy.txt", f"{data.BASE_URL}/toy.meta"]
    assert (txt.read_bytes(), meta.read_bytes()) == (files["toy.txt"], files["toy.meta"])
    with pytest.raises(ValueError, match="Unknown dataset 'R9'"):
        data.download("R9", tmp_path)


def test_download_verifies_checksum(tmp_path, monkeypatch):
    """A file whose checksum matches is saved; a mismatch raises and leaves nothing."""
    payload = b"0\ttrain\tearn\n"
    sha256 = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: _FakeResponse(payload))

    ok = data.Datasets(file_url="https://example.invalid/R8.meta", path=str(tmp_path),
                       file_name="R8.meta", sha256=sha256)
    assert open(ok.download(), "rb").read() == payload

    bad = data.Datasets(file_url="https://example.invalid/R8.txt", path=str(tmp_path),
                        file_name="R8.txt", sha256="0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        bad.download()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["R8.meta"]


def test_validate_files_with_success():
    """Test file validation with valid input files."""
    # Initialize logger and file operations
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    
    # Validate that text and meta files have matching line counts
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_valid.txt", path_meta="tests/data/file_valid.meta")
    assert num_lines_f1 == num_lines_f2

    # Validate that meta file has correct column structure
    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_valid.meta")
    assert meta_validation == True

    # Perform complete file validation
    message, status = flops.validate_files(path_txt="tests/data/file_valid.txt", 
                                path_meta="tests/data/file_valid.meta")
    
    # Verify validation succeeded and message indicates success
    assert status == True
    assert message == '[FILES VALIDATION] - [meta] Validated file - [corpus] Validated file'


def test_validate_files_fail_meta_missing_column():
    """Test file validation failure when meta file has incorrect column structure."""
    # Initialize logger and file operations
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    
    # Validate line counts (should match even if columns are wrong)
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_valid.txt", path_meta="tests/data/file_meta_2_cols.meta")
    assert num_lines_f1 == num_lines_f2

    # Validate meta columns - should fail because file has only 2 columns instead of expected format
    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_meta_2_cols.meta")
    assert meta_validation == False

    # Perform complete file validation - should fail due to meta file structure
    message, status = flops.validate_files(path_txt="tests/data/file_valid.txt", 
                                path_meta="tests/data/file_meta_2_cols.meta")
    
    # Verify validation failed with appropriate error message
    assert status == False
    assert message == '[FILES VALIDATION] - [meta] The meta file does not correspond to standard - [corpus] Validated file'


def test_validate_files_fail_different_rows_count():
    """Test file validation failure when text and meta files have different row counts."""
    # Initialize logger and file operations
    lgg = logger.PrintLog()
    flops = flop.FileOps(logger=lgg)
    
    # Validate line counts - should detect mismatch (10 lines in txt vs 20 in meta)
    num_lines_f1, num_lines_f2 = flops.validate_input_files_lines(
        path_txt="tests/data/file_txt_10_lines.txt", path_meta="tests/data/file_valid.meta")
    
    # Verify the mismatch is detected
    assert num_lines_f1 == 10
    assert num_lines_f2 == 20
    assert num_lines_f1 < num_lines_f2

    # Meta file structure is valid, but line count doesn't match
    meta_validation = flops.validate_input_meta_columns(
        file_path="tests/data/file_valid.meta")
    assert meta_validation == True

    # Perform complete file validation - should fail due to line count mismatch
    message, status = flops.validate_files(path_txt="tests/data/file_txt_10_lines.txt", 
                                path_meta="tests/data/file_valid.meta")
    
    # Verify validation failed with appropriate error message
    assert status == False
    assert message == '[FILES VALIDATION] - [meta] Validated file - [corpus] Files do not have the same count of lines'


def test_save_corpus_writes_aligned_utf8_files(tmp_path):
    txt, meta = data.save_corpus(
        tmp_path / "out", "toy",
        texts=["first\n line\twith  spaces", "مرحبا بالعالم"],
        labels=["a", "b"], splits=["train", "test"])

    assert txt.read_text(encoding="utf-8") == "first line with spaces\nمرحبا بالعالم\n"
    assert meta.read_text(encoding="utf-8") == "0\ttrain\ta\n1\ttest\tb\n"
    flops = flop.FileOps(logger=logger.PrintLog())
    assert flops.validate_files(path_txt=str(txt), path_meta=str(meta))[1]


@pytest.mark.parametrize("kwargs, message", [
    (dict(texts=["a"], labels=["x", "y"], splits=["train"]), "same length"),
    (dict(texts=["a", "b"], labels=["x", "y"], splits=["train", "dev"]), "'train' or 'test'"),
    (dict(texts=["a", "b"], labels=["x\ty", "y"], splits=["train", "test"]), "tabs or newlines"),
    (dict(texts=["a", "b"], labels=["x", "y"], splits=["train", "train"]), "at least one"),
])
def test_save_corpus_rejects_invalid_input(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        data.save_corpus(tmp_path, "toy", **kwargs)


def test_twenty_newsgroups_keeps_split_and_drops_short_posts(tmp_path, monkeypatch):
    import sklearn.datasets
    from types import SimpleNamespace

    posts = {
        "train": ["a long enough training post", "too short"],
        "test": ["a long enough test post here"],
    }

    def fake_fetch(subset, categories, remove):
        return SimpleNamespace(data=posts[subset], target=[0] * len(posts[subset]),
                               target_names=["sci.med"])

    monkeypatch.setattr(sklearn.datasets, "fetch_20newsgroups", fake_fetch)
    _, meta = data.twenty_newsgroups(tmp_path, name="NG", categories=["sci.med"])

    assert meta.read_text(encoding="utf-8") == "0\ttrain\tsci.med\n1\ttest\tsci.med\n"


def test_load_corpus_meta_accepts_training_split(tmp_path):
    """Regression: Ohsumed uses 'training'; those documents were dropped from both splits."""
    meta = tmp_path / "ds.meta"
    meta.write_text("0\ttraining\tC01\n1\ttrain\tC02\n2\t20news-bydate-train\tC01\n3\ttest\tC02\n",
                    encoding="utf-8")
    flops = flop.FileOps(logger=logger.PrintLog())

    all_docs, train, test = flops.load_corpus_meta(str(meta))

    assert [d.split("\t")[0] for d in train] == ["0", "1", "2"]
    assert [d.split("\t")[0] for d in test] == ["3"]

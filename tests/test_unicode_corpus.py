"""Non-ASCII corpora (GitHub issue #1: Arabic dataset / UTF-8 encoding)."""

import random

import pytest

from text4gcn import GNN, TextPipeline
from text4gcn.builder import Frequency
from text4gcn.modules import clean_data as clean

ARABIC_WORDS = {
    "sport": ["كُرَةُ", "فريق", "مباراة", "لاعب", "هدف", "ملعب"],
    "economy": ["سوق", "اقتصاد", "شركة", "أسهم", "بنك", "تجارة"],
}


@pytest.fixture
def offline_nltk(monkeypatch):
    """Keep the pipeline hermetic: no NLTK downloads, fixed stop-word list."""
    monkeypatch.setattr(clean, "config_nltk", lambda: None)
    monkeypatch.setattr(clean, "retrieve_stop_words", lambda language="english": {"في", "من"})


def _write_corpus(path, name):
    rng = random.Random(0)
    docs, meta = [], []
    for i in range(40):
        label = "sport" if i % 2 else "economy"
        words = [rng.choice(ARABIC_WORDS[label]) for _ in range(12)] + ["في", "من"]
        docs.append(" ".join(words) + "،")  # Arabic comma is stripped
        meta.append(f"{i}\t{'train' if i < 30 else 'test'}\t{label}")
    (path / f"{name}.txt").write_text("\n".join(docs) + "\n", encoding="utf-8")
    (path / f"{name}.meta").write_text("\n".join(meta) + "\n", encoding="utf-8")


def test_clean_str_keeps_non_ascii_words_and_diacritics():
    assert clean.clean_str("كُرَةُ القدم، São_Paulo!", ascii_only=False) == "كُرَةُ القدم são paulo !"
    # Default keeps the original TextGCN behaviour (ASCII only).
    assert clean.clean_str("São Paulo!") == "s o paulo !"


def test_arabic_corpus_end_to_end(tmp_path, offline_nltk):
    _write_corpus(tmp_path, "AR")
    TextPipeline(dataset_name="AR", dataset_path=str(tmp_path), rare_count=1,
                 language="arabic", ascii_only=False).execute()

    vocab = (tmp_path / "AR.shuffled" / "AR.vocab").read_text(encoding="utf-8").split()
    assert set(vocab) == {w for words in ARABIC_WORDS.values() for w in words}

    Frequency(dataset_name="AR", dataset_path=str(tmp_path)).tfidf()
    result = GNN(dataset_name="AR", dataset_path=str(tmp_path), log_dir=str(tmp_path / "log"),
                 epochs=3, device="cpu").fit()
    assert 0.0 <= result.test_accuracy <= 1.0


def test_wrong_encoding_reports_how_to_fix(tmp_path, offline_nltk):
    (tmp_path / "L1.txt").write_bytes("café crème\n".encode("latin1"))
    (tmp_path / "L1.meta").write_text("0\ttrain\ta\n", encoding="utf-8")

    with pytest.raises(UnicodeDecodeError, match="encoding='latin1'"):
        TextPipeline(dataset_name="L1", dataset_path=str(tmp_path)).execute()

    TextPipeline(dataset_name="L1", dataset_path=str(tmp_path), rare_count=1,
                 encoding="latin1").clean_data()
    cleaned = (tmp_path / "L1.cleaned" / "L1.txt").read_text(encoding="utf-8")
    assert cleaned.split() == ["caf", "cr", "me"]

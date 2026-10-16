"""Console output (verbose), diagnostics logging and the data returned by GNN.fit()."""

import logging

import pytest

import text4gcn
from text4gcn import GNN, TextPipeline
from text4gcn.builder import Frequency
from text4gcn.datasets import save_corpus
from text4gcn.modules import clean_data as clean
from text4gcn.utils.reporting import Reporter, format_duration

TOPICS = {
    "space": ["rocket", "orbit", "planet", "moon", "galaxy", "stars", "telescope"],
    "cooking": ["bread", "oven", "garlic", "onion", "soup", "chef", "herbs"],
}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A tiny two-topic corpus; NLTK is kept offline."""
    monkeypatch.setattr(clean, "config_nltk", lambda: None)
    monkeypatch.setattr(clean, "retrieve_stop_words", lambda language="english": set())
    texts, labels, splits = [], [], []
    for i in range(24):
        topic = "space" if i % 2 else "cooking"
        words = TOPICS[topic]
        texts.append(" ".join(words[(i + k) % len(words)] for k in range(5)))
        labels.append(topic)
        splits.append("train" if i < 18 else "test")
    save_corpus(tmp_path, "toy", texts=texts, labels=labels, splits=splits)
    return tmp_path


@pytest.mark.parametrize("seconds, text", [(0.001, "<0.1 s"), (0.42, "0.4 s"), (3.14, "3.1 s"),
                                           (42.4, "42 s"), (245, "4m05s"), (3725, "1h02m")])
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


def test_reporter_rejects_unknown_verbose():
    with pytest.raises(ValueError, match="verbose"):
        Reporter(verbose=3)


def test_silent_reporter_still_keeps_the_log(tmp_path, capsys):
    report = Reporter(verbose=0)
    report.header("something")
    with report.step("stage") as step:
        step.summary = "done"
    report.save(tmp_path / "log.txt")

    assert capsys.readouterr().out == ""
    log = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "text4gcn · something" in log and "✓ stage" in log


def test_verbose_0_prints_nothing(corpus, capsys):
    TextPipeline(dataset_name="toy", dataset_path=corpus, rare_count=1, verbose=0).execute()
    Frequency(dataset_name="toy", dataset_path=corpus, verbose=0).tfidf()
    GNN(dataset_name="toy", dataset_path=corpus, log_dir=str(corpus / "logs"), epochs=5,
        device="cpu", verbose=0).fit()
    assert capsys.readouterr().out == ""


def test_verbose_2_prints_one_line_per_epoch(corpus, capsys):
    TextPipeline(dataset_name="toy", dataset_path=corpus, rare_count=1, verbose=0).execute()
    Frequency(dataset_name="toy", dataset_path=corpus, verbose=0).tfidf()
    result = GNN(dataset_name="toy", dataset_path=corpus, log_dir=str(corpus / "logs"), epochs=5,
                 device="cpu", verbose=2).fit()
    out = capsys.readouterr().out
    assert out.count("    epoch ") == result.epochs_trained
    assert "✓ train" in out and "✓ test" in out


def test_fit_returns_history_metrics_and_named_classes(corpus):
    TextPipeline(dataset_name="toy", dataset_path=corpus, rare_count=1, verbose=0).execute()
    Frequency(dataset_name="toy", dataset_path=corpus, verbose=0).tfidf()
    result = GNN(dataset_name="toy", dataset_path=corpus, log_dir=str(corpus / "logs"), epochs=5,
                 device="cpu", verbose=0).fit()

    assert len(result.history) == result.epochs_trained
    assert list(result.history.columns) == ["loss", "accuracy", "val_loss", "val_accuracy", "seconds"]
    assert 1 <= result.best_epoch <= result.epochs_trained
    assert result.metrics["accuracy"] == pytest.approx(result.test_accuracy)
    assert set(result.report.index) == {"space", "cooking"}
    assert list(result.confusion_matrix.index) == list(result.confusion_matrix.columns)
    assert int(result.confusion_matrix.to_numpy().sum()) == 6


def test_pipeline_writes_the_processing_log(corpus):
    TextPipeline(dataset_name="toy", dataset_path=corpus, rare_count=1, verbose=0).execute()
    Frequency(dataset_name="toy", dataset_path=corpus, verbose=0).tfidf()
    log = (corpus / "log" / "toy_dataset.txt").read_text(encoding="utf-8")
    assert "✓ clean" in log and "✓ features" in log and 'Frequency.tfidf("toy")' in log


def test_set_verbosity_shows_diagnostics(corpus, caplog):
    text4gcn.set_verbosity("debug")
    try:
        with caplog.at_level(logging.DEBUG, logger="text4gcn"):
            TextPipeline(dataset_name="toy", dataset_path=corpus, rare_count=1, verbose=0).execute()
        assert any("allx" in record.getMessage() for record in caplog.records)
    finally:
        text4gcn.set_verbosity("warning")


def test_set_verbosity_rejects_unknown_level():
    with pytest.raises(ValueError, match="Unknown verbosity"):
        text4gcn.set_verbosity("chatty")

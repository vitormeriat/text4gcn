"""Input format and the errors raised when a pipeline step is missing."""

import pytest

from text4gcn import GNN, Builder, TextPipeline
from text4gcn.builder import Frequency
from text4gcn.datasets import save_corpus

TEXTS = ["sweet ripe apple pie", "sweet ripe banana bread", "fast red car engine",
         "fast blue car wheel"] * 3


@pytest.fixture
def corpus(tmp_path):
    save_corpus(tmp_path, "toy", texts=TEXTS, labels=["sweet fruit", "fast car"] * 6,
                splits=["train"] * 8 + ["test"] * 4)
    return tmp_path


def test_labels_may_contain_spaces(corpus):
    """Regression: labels were split on any whitespace, so 'sweet fruit' broke TextPipeline."""
    TextPipeline(dataset_name="toy", dataset_path=corpus, rare_count=1).execute()
    Frequency(dataset_name="toy", dataset_path=corpus, verbose=0).tfidf()

    result = GNN(dataset_name="toy", dataset_path=corpus, log_dir=str(corpus / "log"),
                 epochs=2, device="cpu", verbose=0).fit()

    assert set(result.report.index) >= {"sweet fruit", "fast car"}


def test_builder_before_preprocessing_shows_the_fix(corpus):
    with pytest.raises(FileNotFoundError, match=r"TextPipeline\("):
        Frequency(dataset_name="toy", dataset_path=corpus, verbose=0).tfidf()


def test_builder_rejects_missing_path_and_empty_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset_path does not exist"):
        Frequency(dataset_name="toy", dataset_path=tmp_path / "missing")
    with pytest.raises(ValueError, match="dataset_name"):
        Frequency(dataset_name=" ", dataset_path=tmp_path)


def test_gnn_names_the_builder_of_a_missing_graph(corpus):
    """The adjacency directory exists, but not the file of the requested builder."""
    TextPipeline(dataset_name="toy", dataset_path=corpus, rare_count=1).execute()
    Frequency(dataset_name="toy", dataset_path=corpus, verbose=0).tfidf()

    gnn = GNN(dataset_name="toy", dataset_path=corpus, builder=Builder.Bert, device="cpu")
    with pytest.raises(FileNotFoundError, match=r"ind\.embedding\.bert\.toy\.adj[\s\S]*\.bert\(\)"):
        gnn.fit()

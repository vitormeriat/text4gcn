"""Regression tests for the word-word edge builders: Word2Vec RWF, cosine similarity, LIWC."""

import pickle
import subprocess
import sys

import numpy as np
import pytest

from text4gcn.builder.linguistic_inquiry_adj import LiwcAdjacency
from text4gcn.modules.adjacency import extract_cosine_similarity_word_weights
from text4gcn.modules.textvec.word_features import train_word2vec
from text4gcn.modules.word_processor import extract_tf_idf_word_vectors

CORPUS = [["rocket", "orbit", "planet", "moon"], ["bread", "oven", "bake", "crust"]] * 20


class TestWord2Vec:
    def test_num_epochs_reaches_gensim(self):
        """Regression: num_epochs was never passed on, so gensim always trained 5 epochs."""
        model = train_word2vec(None, CORPUS, num_epochs=7, embedding_dimension=8, training_regime=1)
        assert model.epochs == 7

    def test_same_seed_gives_same_vectors(self):
        first = train_word2vec(None, CORPUS, num_epochs=3, embedding_dimension=8, training_regime=1, seed=7)
        second = train_word2vec(None, CORPUS, num_epochs=3, embedding_dimension=8, training_regime=1, seed=7)
        np.testing.assert_array_equal(first.wv["orbit"], second.wv["orbit"])

    def test_vectors_do_not_depend_on_python_hash_seed(self):
        """gensim seeds each vector with hash(word); str hashing changes with PYTHONHASHSEED."""
        script = (
            "from text4gcn.modules.textvec.word_features import train_word2vec\n"
            f"m = train_word2vec(None, {CORPUS!r}, num_epochs=2, embedding_dimension=8, training_regime=1)\n"
            "print(repr(float(m.wv['orbit'].sum())))"
        )
        outputs = {
            subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": hash_seed}).stdout.strip().splitlines()[-1]
            for hash_seed in ("1", "2")
        }
        assert len(outputs) == 1


class TestCosineSimilarityVectors:
    def test_words_without_definition_get_zero_vectors(self):
        vectors = extract_tf_idf_word_vectors(["a large star", "", "a small star"], max_features=10)
        assert not vectors[1].any()
        assert vectors[0].any() and vectors[2].any()

    def test_no_definitions_at_all_does_not_raise(self):
        """A non-English vocabulary has no WordNet definitions."""
        vectors = extract_tf_idf_word_vectors(["", ""], max_features=10)
        assert vectors.shape[0] == 2 and not vectors.any()

    def test_words_without_definition_get_no_edges(self, tmp_path):
        """Regression: every undefined word got the same '<PAD>' vector, forming a clique."""
        vocab = ["star", "sun", "xq", "zz"]
        vectors = extract_tf_idf_word_vectors(["a luminous star", "the star at the centre", "", ""], 10)
        path = tmp_path / "vectors.pkl"
        path.write_bytes(pickle.dumps(dict(zip(vocab, vectors))))
        rows, cols, _ = extract_cosine_similarity_word_weights(vocab, train_size=0, word_vec_path=str(path),
                                                               threshold=0.1)
        assert {2, 3}.isdisjoint(rows) and {2, 3}.isdisjoint(cols)
        assert (0, 1) in set(zip(rows, cols))


LIWC_DIC = "%\n1\tposemo\n2\tnegemo\n3\tfood\n%\nhappy\t1\njoy*\t1\nsad\t2\nbread\t3\ncake\t3\n"


class TestLiwc:
    @pytest.fixture
    def builder(self, tmp_path):
        dic = tmp_path / "toy.dic"
        dic.write_text(LIWC_DIC, encoding="utf-8")
        return LiwcAdjacency(dataset_name="toy", dataset_path=str(tmp_path), liwc_path=str(dic))

    def test_adjacent_words_sharing_a_category_are_paired(self, builder):
        """Regression: words were looked up letter by letter, so real words matched nothing."""
        pairs = builder.build_relation_pair([["happy", "joyful", "sad", "bread", "cake"]])
        assert set(pairs) == {"happy,joyful", "joyful,happy", "bread,cake", "cake,bread"}

    def test_pairs_are_counted_per_occurrence(self, builder):
        pairs = builder.build_relation_pair([["bread", "cake"], ["cake", "bread"], ["bread", "sad"]])
        assert pairs == {"bread,cake": 2, "cake,bread": 2}

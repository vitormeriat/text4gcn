from gensim.models import Word2Vec
from typing import List
import os
import zlib


def _stable_hash(text: str) -> int:
    """Process-independent hash, so that gensim's vector initialisation does not
    depend on PYTHONHASHSEED (the built-in ``hash`` of a str is randomised)."""
    return zlib.crc32(text.encode("utf-8"))


def train_word2vec(
    save_dir: str,
    document_list: List[List[str]],
    num_epochs: int,
    embedding_dimension: int,
    training_regime: int,
    seed: int = 1234,
    workers: int = 1,
) -> Word2Vec:
    """Train Word2Vec on the corpus.

    Training regime: 1 for skip-gram, otherwise (0) CBOW.

    With ``workers=1`` the result is reproducible for a given ``seed``; more
    workers train faster, but thread scheduling makes the vectors vary between runs.
    """
    print('[INFO] Training Word2Vec...')
    model = Word2Vec(document_list,
                     vector_size=embedding_dimension,
                     window=5,
                     workers=workers,
                     sg=training_regime,
                     min_count=1,
                     epochs=num_epochs,
                     seed=seed,
                     hashfxn=_stable_hash)

    if save_dir is not None:
        model.save(os.path.join(save_dir, 'word2vec.model'))

    return model

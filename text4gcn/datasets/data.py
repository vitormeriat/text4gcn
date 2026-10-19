import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import requests

# Benchmark corpora published as assets of the `datasets-v1` GitHub release.
BASE_URL = "https://github.com/vitormeriat/text4gcn/releases/download/datasets-v1"

# name -> expected SHA-256 of `<name>.txt` and `<name>.meta`;
# a corrupted or tampered download is rejected.
BENCHMARKS = {
    "R8": ("c2758703ba8b3e29162e8e9f606e9a1f032c408cd9d76380f02c8ba16422cac7",
           "4ab8362c508bf7b1c591bca4e4d72ff6e9ff7a730a94cb0a5dfddf237a7cc025"),
    "R52": ("926453a3a242f0fe61a1a99ab1d3e4e193daf3bc926f3b61b9fc77718c137efc",
            "dd0c08ca54cb51be909f309525643fdc235efecf674a91d854306b939badd239"),
    "ohsumed": ("c117cc9f36c3c121793f85955b47fba1bbb58a8eb051c638e3611544f11f5032",
                "724b8ed25eb3d99b90b86fdc4ef5e2450b386ff0ba348afe38f420cd61eccc16"),
    "pubmed": ("71a54036f68f4f0de2680d7c84f0a015588bf0309ea891ae272d55714795b233",
               "daed1806b5b369f070d66224c90cbf01431c29b98b869a1f4de0db6f13ac64fa"),
}


def list():
    """Names accepted by :func:`download`."""
    return [*BENCHMARKS]


class Datasets():
    def __init__(self, file_url: str, path: str | os.PathLike[str], file_name: str, sha256: str) -> None:
        self.file_url = file_url
        self.path = path
        self.file_name = file_name
        self.sha256 = sha256

    def download(self) -> str:
        """Download the file into `path`, verify its checksum and return its path."""
        absp = os.path.abspath(os.path.expanduser(self.path))
        os.makedirs(absp, exist_ok=True)
        file_path = os.path.join(absp, self.file_name)

        r = requests.get(self.file_url, stream=True, timeout=60)
        r.raise_for_status()

        digest = hashlib.sha256()
        tmp_path = f"{file_path}.part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                digest.update(chunk)
                f.write(chunk)

        if digest.hexdigest() != self.sha256:
            os.remove(tmp_path)
            raise ValueError(
                f"Checksum mismatch for {self.file_name} downloaded from {self.file_url}: "
                f"expected {self.sha256}, got {digest.hexdigest()}"
            )
        os.replace(tmp_path, file_path)
        return file_path


def download(name: str, path: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Download a benchmark corpus in the Text4GCN input format.

    Writes `<name>.txt` and `<name>.meta` into `path`, after verifying each
    file against its SHA-256 checksum.

    ===========  =========  ======  =====  =======
    name         documents  train   test   classes
    ===========  =========  ======  =====  =======
    ``R8``       7,674      5,485   2,189  8
    ``R52``      9,100      6,532   2,568  52
    ``ohsumed``  7,400      3,357   4,043  23
    ``pubmed``   19,717     13,802  5,915  3
    ===========  =========  ======  =====  =======

    Parameters
    ----------
    name:
        One of :func:`list`. Use the same value as `TextPipeline(dataset_name=...)`.
    path:
        Directory where the files are saved (created if missing; `~` is expanded).

    Returns
    -------
    (txt_path, meta_path)

    Example
    -------
        >>> from text4gcn.datasets import download
        >>> download("R8", "data")
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(BENCHMARKS)}.")
    txt_sha256, meta_sha256 = BENCHMARKS[name]
    txt_path, meta_path = (
        Path(Datasets(file_url=f"{BASE_URL}/{name}{ext}", path=path,
                      file_name=f"{name}{ext}", sha256=sha256).download())
        for ext, sha256 in ((".txt", txt_sha256), (".meta", meta_sha256))
    )
    return txt_path, meta_path


SPLITS = ("train", "test")


def save_corpus(
    path: str | os.PathLike[str],
    name: str,
    texts: Iterable[str],
    labels: Iterable[str],
    splits: Iterable[str],
) -> tuple[Path, Path]:
    """Write a labelled corpus in the Text4GCN input format.

    Creates `<path>/<name>.txt` (one document per line, whitespace collapsed)
    and `<path>/<name>.meta` (`<index>\\t<train|test>\\t<label>`), both UTF-8.
    This is the input expected by `TextPipeline(dataset_name=name, dataset_path=path)`.

    Parameters
    ----------
    path:
        Output directory (created if missing).
    name:
        Dataset name: the prefix of both files.
    texts, labels, splits:
        One entry per document; `splits` values must be ``"train"`` or ``"test"``.

    Returns
    -------
    (txt_path, meta_path)

    Example
    -------
        >>> from text4gcn.datasets import save_corpus
        >>> save_corpus("data", "reviews",
        ...             texts=["great movie", "boring plot"],
        ...             labels=["pos", "neg"], splits=["train", "test"])
    """
    texts, labels, splits = [*texts], [*labels], [*splits]
    if not len(texts) == len(labels) == len(splits):
        raise ValueError(
            f"texts, labels and splits must have the same length, got "
            f"{len(texts)}, {len(labels)} and {len(splits)}."
        )
    invalid = {s for s in splits if s not in SPLITS}
    if invalid:
        raise ValueError(f"splits must be 'train' or 'test', got {sorted(invalid)}.")
    bad_labels = {str(lb) for lb in labels if re.search(r"[\t\n\r]", str(lb))}
    if bad_labels:
        raise ValueError(f"labels must not contain tabs or newlines: {sorted(bad_labels)}")
    if "train" not in splits or "test" not in splits:
        raise ValueError("the corpus needs at least one 'train' and one 'test' document.")

    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    txt_path, meta_path = out / f"{name}.txt", out / f"{name}.meta"
    txt_path.write_text(
        "".join(re.sub(r"\s+", " ", t).strip() + "\n" for t in texts), encoding="utf-8")
    meta_path.write_text(
        "".join(f"{i}\t{s}\t{lb}\n" for i, (s, lb) in enumerate(zip(splits, labels))),
        encoding="utf-8")
    return txt_path, meta_path


def twenty_newsgroups(
    path: str | os.PathLike[str],
    name: str = "20NG",
    categories: Sequence[str] | None = None,
    min_words: int = 5,
    remove: Sequence[str] = ("headers", "footers", "quotes"),
) -> tuple[Path, Path]:
    """Download 20 Newsgroups (via scikit-learn) and save it with :func:`save_corpus`.

    Keeps the official train/test split. `remove` strips e-mail headers,
    signatures and quoted replies so the model learns from the message body;
    documents with fewer than `min_words` words afterwards are dropped.

    Parameters
    ----------
    path:
        Output directory.
    name:
        Dataset name (file prefix). Default ``"20NG"``.
    categories:
        Subset of newsgroups, e.g. ``["sci.med", "comp.graphics"]``.
        ``None`` loads all 20 (18,846 documents: needs a large GPU, as the
        whole corpus becomes a single graph).
    min_words:
        Minimum number of words to keep a document.
    remove:
        Parts removed by :func:`sklearn.datasets.fetch_20newsgroups`.

    Example
    -------
        >>> from text4gcn.datasets import twenty_newsgroups
        >>> twenty_newsgroups("data", name="20NG4",
        ...                   categories=["alt.atheism", "sci.med"])
    """
    from sklearn.datasets import fetch_20newsgroups

    texts, labels, splits = [], [], []
    for split in SPLITS:
        bunch = fetch_20newsgroups(subset=split, categories=categories, remove=tuple(remove))
        for text, target in zip(bunch.data, bunch.target):
            if len(text.split()) >= min_words:
                texts.append(text)
                labels.append(bunch.target_names[target])
                splits.append(split)
    return save_corpus(path, name, texts, labels, splits)

# Datasets

```python
from text4gcn.datasets import download, save_corpus, twenty_newsgroups
```

Helpers that write a corpus in the Text4GCN input format: two aligned UTF-8 files in one directory.

| File | One line per document |
|---|---|
| `<name>.txt` | the document text, on a single line |
| `<name>.meta` | `<doc id>\t<train|test>\t<label>` |

Pass the same directory and name to `TextPipeline(dataset_name=name, dataset_path=path)`.

---

## save_corpus

```python
def save_corpus(
    path: str | os.PathLike[str],
    name: str,
    texts: Iterable[str],
    labels: Iterable[str],
    splits: Iterable[str],
) -> tuple[Path, Path]
```

Writes your own labelled corpus. Line breaks and tabs inside each text are collapsed to single spaces.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| PathLike` | Output directory (created if missing). |
| `name` | `str` | Dataset name: the prefix of both files. |
| `texts` | `Iterable[str]` | One document per entry. |
| `labels` | `Iterable[str]` | Class of each document. Must not contain tabs or newlines. |
| `splits` | `Iterable[str]` | `"train"` or `"test"` for each document. Both must be present. |

**Returns** the paths of the `.txt` and `.meta` files.

**Raises** `ValueError` when the three inputs have different lengths, a split is not `train`/`test`, a label contains a tab or newline, or one of the splits is missing.

```python
import pandas as pd
from text4gcn.datasets import save_corpus

df = pd.read_csv("reviews.csv")          # columns: text, label, split
save_corpus("data", "reviews", texts=df.text, labels=df.label, splits=df.split)
```

See [5 · Bring your own corpus](../tutorials/05_bring_your_own_corpus.ipynb) for a walkthrough, including non-English text.

---

## twenty_newsgroups

```python
def twenty_newsgroups(
    path: str | os.PathLike[str],
    name: str = "20NG",
    categories: Sequence[str] | None = None,
    min_words: int = 5,
    remove: Sequence[str] = ("headers", "footers", "quotes"),
) -> tuple[Path, Path]
```

Downloads [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) with scikit-learn, keeps the official train/test split, and saves it with `save_corpus`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| PathLike` | — | Output directory. |
| `name` | `str` | `"20NG"` | Dataset name (file prefix). |
| `categories` | `Sequence[str] \| None` | `None` | Subset of newsgroups. `None` loads all 20 (18,846 documents), which needs a large GPU because the whole corpus becomes one graph. |
| `min_words` | `int` | `5` | Documents with fewer words (after `remove`) are dropped. |
| `remove` | `Sequence[str]` | headers, footers, quotes | Parts stripped by `fetch_20newsgroups`, so the model learns from the message body. |

```python
from text4gcn.datasets import twenty_newsgroups

twenty_newsgroups("data", name="20NG4",
                  categories=["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"])
```

---

## download

```python
def download(name: str, path: str | os.PathLike[str]) -> tuple[Path, Path]
```

Downloads a benchmark corpus into `path` (created if missing; `~` is expanded) as `<name>.txt` / `<name>.meta`. Each file is downloaded to a temporary `.part` file and verified against a SHA-256 checksum before it replaces `<name>.txt` / `<name>.meta`.

**Returns** the paths of the `.txt` and `.meta` files.

**Raises** `ValueError` for a `name` not in `list()`, or on a checksum mismatch (only the failing `.part` file is deleted; a `.txt` already verified stays); `requests.HTTPError` when the download fails with an HTTP error.

| `name` | Corpus | Documents | Train | Test | Classes |
|---|---|---|---|---|---|
| `"R8"` | Reuters-21578, 8 classes | 7,674 | 5,485 | 2,189 | 8 |
| `"R52"` | Reuters-21578, 52 classes | 9,100 | 6,532 | 2,568 | 52 |
| `"ohsumed"` | MEDLINE cardiovascular abstracts | 7,400 | 3,357 | 4,043 | 23 |
| `"pubmed"` | PubMed diabetes abstracts | 19,717 | 13,802 | 5,915 | 3 |

```python
from text4gcn import TextPipeline
from text4gcn.datasets import download

download("ohsumed", "data")
TextPipeline(dataset_name="ohsumed", dataset_path="data").execute()
```

`text4gcn.datasets.data.list()` returns the accepted names. Use the same `name` as `dataset_name` in the next steps.

!!! warning "Size"
    PubMed (19.7k documents) produces a large graph: plan for roughly `(documents + words)² × 20 bytes` of memory during training (see [tutorial 5](../tutorials/05_bring_your_own_corpus.ipynb)).

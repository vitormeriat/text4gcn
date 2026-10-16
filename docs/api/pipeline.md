# TextPipeline

```python
from text4gcn import TextPipeline
```

Cleans and shuffles the raw corpus, builds the vocabulary, and writes the label matrices and train / validation / test split that `GNN` loads. (It also writes node feature matrices for compatibility with the TextGCN file layout; the GCN itself uses one-hot node features.)

---

## TextPipeline

```python
class TextPipeline:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        rare_count: int = 5,
        language: str = "english",
        seed: int = 1234,
        encoding: str = "utf-8",
        ascii_only: bool = True,
        verbose: int = 1,
    ) -> None
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_name` | `str` | — | Dataset name. Must match the `.txt` and `.meta` file prefixes. |
| `dataset_path` | `str` | — | Directory containing the raw dataset files. Must exist at construction time. |
| `rare_count` | `int` | `5` | Words appearing fewer than this many times across the corpus are dropped. |
| `language` | `str` | `"english"` | Language for stop-word removal (NLTK stop-word list). |
| `seed` | `int` | `1234` | Random seed for document shuffle (reproducible train/test order). |
| `encoding` | `str` | `"utf-8"` | Encoding of the input `.txt` corpus. Use `"latin1"` for legacy corpora that are not valid UTF-8. Intermediate files are always UTF-8. |
| `ascii_only` | `bool` | `True` | Keep only `A-Za-z0-9` when cleaning (original TextGCN). Set `False` for Arabic, Cyrillic, accented Latin, etc. |
| `verbose` | `int` | `1` | `0` silent; `1` or `2` one summary line per stage. |

!!! example "Non-English corpus (e.g. Arabic)"

    ```python
    TextPipeline(
        dataset_name="MyArabic",
        dataset_path="data/",
        language="arabic",   # NLTK stop-word list
        ascii_only=False,    # keep Arabic letters and diacritics
    ).execute()
    ```

---

### `execute()`

```python
def execute(
    self,
    validation_ratio: float = 0.10,
    seed: int | None = None,
) -> None
```

Runs the complete preprocessing pipeline. Stages execute in order:

1. **Clean** — remove stop words and rare words, write to `{dataset_name}.cleaned/`
2. **Shuffle** — randomise train/test order, write to `{dataset_name}.shuffled/`
3. **Prepare words** — build vocabulary and TF-IDF word vectors
4. **Build node features** — compute x, y, tx, ty, allx, ally matrices, write to `{dataset_name}.node_features/`

Each stage prints one line, for example:

```text
text4gcn · TextPipeline("20NG4") · language english · validation 10%
  ✓ clean     3,647 documents · 10,890 words kept (rare_count=5)  (1.1 s)
  ✓ shuffle   train 2,192 · test 1,455 documents (seed 1234)  (0.2 s)
  ✓ words     10,890 words · 8,941 with a WordNet definition  (5.7 s)
  ✓ features  train 1,973 · validation 219 · test 1,455 documents · 4 classes  (1.8 s)
```

The same lines, with a timestamp, are written to the processing log
`{dataset_path}/log/{dataset_name}_dataset.txt`; the graph builders append theirs.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `validation_ratio` | `float` | `0.10` | Fraction of the training set held out for validation (DCAI 2022 / Yao et al. use `0.10`). This is the only place where the validation split is set. |
| `seed` | `int \| None` | `None` | Overrides the shuffle seed given to `TextPipeline(seed=…)` for this run. |

---

## Output layout

After `execute()` the dataset directory contains:

```
data/
  R8.txt                     ← original input (unchanged)
  R8.meta                    ← original input (unchanged)
  R8.cleaned/
    R8.txt                   ← stop-word and rare-word filtered
  R8.shuffled/
    R8.txt                   ← shuffled corpus
    R8.meta                  ← shuffled metadata
    R8.train                 ← training document indices
    R8.test                  ← test document indices
    R8.vocab                 ← vocabulary (one word per line)
    R8.word_vectors          ← TF-IDF word vectors (pickle)
  R8.node_features/
    ind.R8.x                 ← train doc feature matrix
    ind.R8.y                 ← train doc one-hot labels
    ind.R8.tx                ← test doc feature matrix
    ind.R8.ty                ← test doc one-hot labels
    ind.R8.allx              ← train + word node features
    ind.R8.ally              ← train + word node labels
  log/
    R8_dataset.txt           ← pipeline execution log
```

---

## Examples

**Basic usage:**

```python
from text4gcn import TextPipeline

TextPipeline(
    dataset_name="R8",
    dataset_path="data/",
    seed=1234,
).execute()
```

**Custom rare-word threshold and validation split:**

```python
TextPipeline(
    dataset_name="R8",
    dataset_path="data/",
    rare_count=3,          # keep words appearing ≥ 3 times
    language="english",
    seed=1234,
).execute(
    validation_ratio=0.15, # use 15% of train for validation
)
```

**French corpus:**

```python
TextPipeline(
    dataset_name="lemonde",
    dataset_path="data/",
    language="french",
).execute()
```

---

## Input file format

**`{dataset_name}.txt`** — one document per line:

```
computer software programming algorithm data structures
natural language processing text classification neural
```

**`{dataset_name}.meta`** — tab-separated `index  split  label`:

```
0	train	technology
1	train	science
2	test	technology
3	test	business
```

!!! note "Split values"
    `train`, `training`, and `20news-bydate-train` are treated as training documents. All other values are treated as test documents.

from typing import Any, Dict, Iterable, Optional
from collections import Counter
from nltk import word_tokenize
from typing import List

def flatten_nested_iterables(iterables_of_iterables: Iterable[Iterable[Any]]) -> Iterable[Any]:
    return [item for sublist in iterables_of_iterables for item in sublist]
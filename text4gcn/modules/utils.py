from typing import Any, Iterable


def flatten_nested_iterables(iterables_of_iterables: Iterable[Iterable[Any]]) -> Iterable[Any]:
    return [item for sublist in iterables_of_iterables for item in sublist]

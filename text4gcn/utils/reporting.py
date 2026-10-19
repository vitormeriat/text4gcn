"""Console output of the pipeline, the graph builders and training.

Every step that runs for a while takes a ``verbose`` argument, as in Keras:

* ``0``: silent;
* ``1`` (default): one summary line per stage, and progress bars while it runs;
* ``2``: the summary lines, and one line per training epoch instead of a bar
  (stable output for log files, CI and saved notebooks).

The same summary lines are appended to the dataset's processing log,
``{dataset_path}/log/{dataset_name}_dataset.txt``, whatever ``verbose`` is.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def format_duration(seconds: float) -> str:
    """``<0.1 s``, ``0.4 s``, ``3.1 s``, ``42 s``, ``4m05s``, ``1h02m``."""
    if seconds < 0.05:
        return "<0.1 s"
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_count(n: int) -> str:
    """``14,537``; ``4.53 M`` from a million up."""
    return f"{n / 1e6:.2f} M" if n >= 1_000_000 else f"{n:,}"


def format_bytes(n: int) -> str:
    for unit, size in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def check_verbose(verbose: int) -> int:
    if verbose not in (0, 1, 2):
        raise ValueError(f"verbose must be 0, 1 or 2, got {verbose!r}.")
    return verbose


class _Step:
    """What a running stage reports when it finishes (set ``summary``)."""

    def __init__(self) -> None:
        self.summary = ""


class Reporter:
    """Prints the summary lines of one run and keeps them for the processing log."""

    def __init__(self, verbose: int = 1, stream=None) -> None:
        self.verbose = check_verbose(verbose)
        self.stream = stream if stream is not None else sys.stdout
        self.lines: list[str] = []

    def write(self, text: str = "") -> None:
        self.lines.append(text)
        if self.verbose:
            print(text, file=self.stream, flush=True)

    def header(self, text: str) -> None:
        self.write(f"text4gcn · {text}")

    @contextmanager
    def step(self, name: str) -> Iterator[_Step]:
        """Time a stage and print ``✓ name  summary  (duration)`` when it ends."""
        step = _Step()
        start = perf_counter()
        yield step
        self.write(f"  ✓ {name:<9} {step.summary}  ({format_duration(perf_counter() - start)})")

    def progress(self, iterable: Iterable[T], desc: str, total: int | None = None,
                 unit: str = "it") -> Iterable[T]:
        """A progress bar over ``iterable`` with ``verbose=1``; the iterable itself otherwise."""
        if self.verbose != 1:
            return iterable
        return tqdm(iterable, desc=f"    {desc}", total=total, unit=unit, file=self.stream,
                    leave=False, dynamic_ncols=True)

    def bar(self, total: int, desc: str, unit: str = "it") -> tqdm | None:
        """A manually updated progress bar with ``verbose=1``; ``None`` otherwise."""
        if self.verbose != 1:
            return None
        return tqdm(total=total, desc=desc, unit=unit, file=self.stream, leave=False,
                    dynamic_ncols=True)

    def save(self, path: str | Path, mode: str = "a") -> None:
        """Append (or write, with ``mode="w"``) the summary lines to a log file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(path, mode, encoding="utf-8") as log:
            log.write(f"# {stamp}\n")
            log.writelines(f"{line}\n" for line in self.lines)
            log.write("\n")

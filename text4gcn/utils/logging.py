"""Diagnostic logging for Text4GCN.

Text4GCN logs diagnostics (array shapes, model structure, file paths) through the
standard :mod:`logging` module under the ``"text4gcn"`` logger. As recommended for
libraries, that logger only has a :class:`logging.NullHandler`: nothing is shown
unless the application configures logging, or calls :func:`set_verbosity`.

Progress and results are not logged here; they are printed by each step according
to its ``verbose`` argument (see :mod:`text4gcn.utils.reporting`).
"""

from __future__ import annotations

import logging
import os

_LIBRARY_LOGGER = "text4gcn"
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
_handler: logging.Handler | None = None


def get_logger(name: str) -> logging.Logger:
    """Logger for a Text4GCN module (``__name__`` of a module inside the package)."""
    return logging.getLogger(name if name.startswith(_LIBRARY_LOGGER) else f"{_LIBRARY_LOGGER}.{name}")


def set_verbosity(level: str | int) -> None:
    """Show Text4GCN diagnostics on stderr from ``level`` up.

    ``level`` is ``"debug"``, ``"info"``, ``"warning"``, ``"error"``, ``"critical"``
    or a :mod:`logging` level number. The environment variable
    ``TEXT4GCN_VERBOSITY`` sets it when the package is imported.

    Example::

        import text4gcn
        text4gcn.set_verbosity("debug")   # shapes, model structure, file paths
    """
    global _handler
    if isinstance(level, str):
        try:
            level = _LEVELS[level.lower()]
        except KeyError:
            raise ValueError(f"Unknown verbosity {level!r}; use one of {sorted(_LEVELS)}.") from None

    logger = logging.getLogger(_LIBRARY_LOGGER)
    logger.setLevel(level)
    if _handler is None:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(_handler)
    _handler.setLevel(level)


def _configure_library_logger() -> None:
    logging.getLogger(_LIBRARY_LOGGER).addHandler(logging.NullHandler())
    env_level = os.environ.get("TEXT4GCN_VERBOSITY")
    if env_level:
        set_verbosity(env_level)

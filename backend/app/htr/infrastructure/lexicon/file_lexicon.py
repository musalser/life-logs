"""Dictionary artifact loaded once per process.

``scripts/build_htr_lexicon.py`` writes a Bloom filter next to the HTR storage;
the backend only reads it. When the file is missing the lexicon is simply
*unavailable*: the UI then shows no OOV marks at all instead of painting every
word of the page as unknown.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from .bloom import BloomFilter

logger = logging.getLogger(__name__)

_loaded: dict[tuple[str, int, int], "FileLexicon"] = {}
_lock = threading.Lock()


class FileLexicon:
    """Read-only word list backed by a Bloom filter artifact."""

    def __init__(self, path: str | Path, bloom: BloomFilter | None = None):
        self.path = str(path)
        self._bloom = bloom

    @property
    def is_available(self) -> bool:
        return self._bloom is not None

    @property
    def word_count(self) -> int:
        return len(self._bloom) if self._bloom is not None else 0

    def __contains__(self, word: object) -> bool:
        if self._bloom is None:
            return False
        return word in self._bloom


def load_file_lexicon(path: str | Path) -> FileLexicon:
    """Load (and cache) the artifact at ``path``.

    A missing file is *not* cached, so dropping a freshly built dictionary next
    to a running backend takes effect on the next request without a restart.
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return FileLexicon(target, None)

    key = (str(target), stat.st_mtime_ns, stat.st_size)
    with _lock:
        cached = _loaded.get(key)
    if cached is not None:
        return cached

    try:
        bloom = BloomFilter.load(target)
    except (OSError, ValueError) as exc:
        logger.warning("HTR lexicon: cannot read %s: %s", target, exc)
        return FileLexicon(target, None)

    lexicon = FileLexicon(target, bloom)
    with _lock:
        for stale in [k for k in _loaded if k[0] == key[0] and k != key]:
            del _loaded[stale]
        _loaded[key] = lexicon
    logger.info(
        "HTR lexicon loaded: %s (%s word forms, %s hash functions)",
        target, bloom.item_count, bloom.hash_count,
    )
    return lexicon


def clear_lexicon_cache() -> None:
    with _lock:
        _loaded.clear()

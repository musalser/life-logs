"""Worker processes for the parts of recognition that hold the GIL.

Profiling one 30-line diary page (3472×4640, RTX 5060 Ti) showed the GPU busy
for ~4 of ~67 seconds; the rest ran on a single core:

* the outline of every line (``calculate_polygonal_environment``) — pure Python
  and shapely, one line after another (≈13 s);
* the prefix beam search over the CTC matrix of every line — pure Python
  (≈27 s before candidate pruning, ≈4 s after).

Threads do not help here: measured, polygonizing with 4 or 8 threads took
*longer* than sequentially (24–25 s against 13 s) because every step holds the
GIL. Both jobs are independent per line, so they go to a pool of processes.

The pool uses the ``spawn`` start method: the parent process has CUDA and a
handful of torch threads initialized, and forking such a process is unsafe. A
spawned worker imports only this module (numpy, and kraken's geometry or the
beam search on first use), never torch's CUDA runtime.

Everything here degrades to in-process execution: with ``workers <= 1``, with
too little work to be worth the transfer, or when the pool breaks (a worker
killed by the OOM killer, for instance). Results are identical either way — the
worker runs the very same function.
"""
from __future__ import annotations

import logging
import os
import threading
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context, shared_memory
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: below this many items a job runs in-process: shipping the work costs more
MIN_PARALLEL_ITEMS = 4

_POOL: ProcessPoolExecutor | None = None
_POOL_LOCK = threading.Lock()
#: 0 = one worker per CPU core; set through :func:`configure`
_CONFIGURED_WORKERS = 0
_POOL_BROKEN = False


def configure(workers: int) -> None:
    """Set the pool size (``0`` = one per CPU core, ``1`` = no pool at all)."""
    global _CONFIGURED_WORKERS
    _CONFIGURED_WORKERS = max(0, int(workers))


def worker_count() -> int:
    if _CONFIGURED_WORKERS > 0:
        return _CONFIGURED_WORKERS
    return max(1, os.cpu_count() or 1)


def _init_worker() -> None:
    """Import what the jobs need and keep every worker on one thread.

    kraken's geometry imports torch, and numpy brings OpenBLAS/OpenMP pools of
    one thread per core: eight workers with eight threads each fight over eight
    cores. Measured on a 30-line page, the outlines took 5.0 s with the default
    thread pools. The imports happen first because a limit only applies to
    libraries that are already loaded.
    """
    import kraken.lib.segmentation  # noqa: F401 - warm import (≈3 s, torch included)

    from .beam import PrefixBeamSearch  # noqa: F401

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - optional
        pass
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - optional
        pass


def _ping(_: Any = None) -> int:
    return os.getpid()


def _pool() -> ProcessPoolExecutor | None:
    global _POOL
    if _POOL_BROKEN or worker_count() <= 1:
        return None
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ProcessPoolExecutor(
                max_workers=worker_count(),
                mp_context=get_context("spawn"),
                initializer=_init_worker,
            )
            # start (and initialize) every worker now rather than on the first
            # page that needs them all
            for _ in range(worker_count()):
                _POOL.submit(_ping)
            logger.info("HTR: started %s CPU worker processes", worker_count())
        return _POOL


def warm_up() -> None:
    """Start the worker processes ahead of the first page (non-blocking)."""
    _pool()


def shutdown() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None


def _run(fn: Callable[[Any], Any], payloads: list[Any]) -> list[Any]:
    """``[fn(p) for p in payloads]``, in the pool when there is one."""
    global _POOL_BROKEN, _POOL
    pool = _pool() if len(payloads) > 1 else None
    if pool is not None:
        try:
            return list(pool.map(fn, payloads))
        except BrokenProcessPool as exc:
            # a dead worker must not fail recognition; the next call runs
            # in-process and the pool is not restarted until the backend is
            logger.warning("HTR: CPU worker pool broke (%s); running in-process", exc)
            with _POOL_LOCK:
                _POOL_BROKEN = True
                _POOL = None
    return [fn(payload) for payload in payloads]


def _chunks(count: int, parts: int) -> list[list[int]]:
    """Interleaved index chunks: long and short lines spread over every worker."""
    parts = max(1, min(parts, count))
    return [list(range(start, count, parts)) for start in range(parts)]


# ---------------------------------------------------------------------------
# Line outlines
# ---------------------------------------------------------------------------


def polygonize(
    im_feats: np.ndarray,
    targets: Sequence[tuple[Sequence, Sequence]],
    topline: bool | None = False,
    raise_on_error: bool = False,
) -> list[Any]:
    """kraken's outline of every ``(baseline, obstacles)`` target, in order.

    Each call is exactly what ``kraken.lib.vgsl.spred.vec_lines`` does for one
    line; the feature map is handed to the workers through shared memory, so it
    is copied once per page instead of once per line.
    """
    targets = list(targets)
    if len(targets) < MIN_PARALLEL_ITEMS or worker_count() <= 1 or _POOL_BROKEN:
        return _polygonize_local(im_feats, targets, topline, raise_on_error)

    features = np.ascontiguousarray(im_feats)
    block = shared_memory.SharedMemory(create=True, size=max(1, features.nbytes))
    try:
        np.ndarray(features.shape, dtype=features.dtype, buffer=block.buf)[...] = features
        payloads = [
            (block.name, features.shape, features.dtype.str, topline, raise_on_error,
             [(index, targets[index][0], targets[index][1]) for index in chunk])
            for chunk in _chunks(len(targets), worker_count())
        ]
        results: list[Any] = [None] * len(targets)
        for chunk_result in _run(_polygonize_chunk, payloads):
            for index, polygon in chunk_result:
                results[index] = polygon
        return results
    finally:
        block.close()
        block.unlink()


def _polygonize_local(im_feats, targets, topline, raise_on_error) -> list[Any]:
    from kraken.lib.segmentation import calculate_polygonal_environment

    return [
        calculate_polygonal_environment(
            baselines=[baseline], im_feats=im_feats, suppl_obj=list(obstacles),
            topline=topline, raise_on_error=raise_on_error,
        )[0]
        for baseline, obstacles in targets
    ]


def _polygonize_chunk(payload) -> list[tuple[int, Any]]:
    name, shape, dtype, topline, raise_on_error, items = payload
    from kraken.lib.segmentation import calculate_polygonal_environment

    block = shared_memory.SharedMemory(name=name)
    try:
        im_feats = np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf)
        out = []
        for index, baseline, obstacles in items:
            polygon = calculate_polygonal_environment(
                baselines=[baseline], im_feats=im_feats, suppl_obj=list(obstacles),
                topline=topline, raise_on_error=raise_on_error,
            )[0]
            # the result must not keep a view into the shared block
            out.append((index, None if polygon is None else np.array(polygon, copy=True)))
        return out
    finally:
        block.close()


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@dataclass
class BeamSpec:
    """Everything a worker needs to rebuild the decoder of the parent.

    The language model travels as a *path* (each worker loads it once and keeps
    it), the vocabulary as the checker itself: the general dictionary pickles to
    its path too (see ``FileLexicon.__reduce__``), so only the author's own words
    are actually copied.
    """

    graphemes: list[str]
    lm_path: str | None
    config: Any
    checker: Any = None


_WORKER_LMS: dict[tuple, Any] = {}


def _worker_lm(path: str | None):
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_mtime_ns, stat.st_size)
    model = _WORKER_LMS.get(key)
    if model is None:
        from ..lm.char_ngram import CharNGram

        _WORKER_LMS.clear()
        model = _WORKER_LMS[key] = CharNGram.load(path)
    return model


class _Codec:
    """The only part of a kraken codec the beam search reads."""

    def __init__(self, graphemes: Sequence[str]):
        # every label but the blank, empty graphemes included: the label count
        # (and with it the grapheme table) must match the parent's exactly
        self.l2c_single = {index: grapheme for index, grapheme in enumerate(graphemes) if index > 0}


def build_decoder(spec: BeamSpec):
    """A :class:`PrefixBeamSearch` equivalent to the parent's one."""
    from .beam import PrefixBeamSearch

    checker = spec.checker
    known = None
    if checker is not None:
        def known(word: str) -> bool:
            return checker.is_known(unicodedata.normalize("NFC", word))

    return PrefixBeamSearch(
        codec=_Codec(spec.graphemes), lm=_worker_lm(spec.lm_path),
        config=spec.config, known_word=known,
    )


def _decode_chunk(payload) -> list[tuple[int, Any]]:
    spec, n, items = payload
    decoder = build_decoder(spec)
    out = []
    for index, frames in items:
        if n <= 1:
            out.append((index, decoder.text_from_frames(frames)))
        else:
            out.append((index, decoder.nbest_from_frames(frames, n)))
    return out


def decode_lines(spec: BeamSpec, frames: Sequence[Any], n: int) -> list[Any]:
    """Beam search over the prepared frames of every line, in order.

    ``frames`` are :meth:`PrefixBeamSearch.frame_candidates` of each line: a few
    candidates per frame instead of the whole ``classes × time`` matrix, so a
    page costs kilobytes to ship rather than tens of megabytes. With ``n <= 1``
    each result is the best text, otherwise the ``n`` best candidates.
    """
    frames = list(frames)
    if len(frames) < MIN_PARALLEL_ITEMS or worker_count() <= 1 or _POOL_BROKEN:
        return [result for _, result in _decode_chunk((spec, n, list(enumerate(frames))))]
    payloads = [
        (spec, n, [(index, frames[index]) for index in chunk])
        for chunk in _chunks(len(frames), worker_count())
    ]
    results: list[Any] = [None] * len(frames)
    for chunk_result in _run(_decode_chunk, payloads):
        for index, value in chunk_result:
            results[index] = value
    return results

"""Compact character n-gram language model for CTC decoding.

The recognizer's codec is a sequence of single code points, and Cyrillic
``й``/``ё`` arrive as *decomposed* sequences (``и`` + U+0306), so the model is
built and queried on the very characters the codec can emit — no
normalization happens between the acoustic model and the language model.

Storage is deliberately flat: every context of every order is packed into a
single ``int64`` key, sorted, and searched with ``np.searchsorted``. A dict of
Python tuples would cost ~100 bytes per context; packing costs eight. That is
what makes a 6-gram over tens of millions of characters fit next to torch in
the application process (tens of megabytes instead of gigabytes).

Scoring is Stupid Backoff: the longest context with an observed continuation
wins, shorter ones are discounted by ``backoff``.

The artifact also carries a ``meta`` dict describing what it was built from. It
is not decoration: the recognizer uses ``meta["running_text_chars"]`` to decide
whether the model is allowed to influence decoding at all. A model built from
word-form lists alone knows letters but has never seen a space in context, and
measured, it decoded *worse* than greedy.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def _bits_for(vocabulary: int) -> int:
    return max(1, int(vocabulary).bit_length())


@dataclass
class _Table:
    """One order: packed contexts -> candidate next characters with log-probs."""

    contexts: np.ndarray  # int64, sorted
    offsets: np.ndarray   # int64, len(contexts) + 1
    values: np.ndarray    # uint32 next character ids
    logprobs: np.ndarray  # float16

    def lookup(self, key: int, char_id: int) -> float | None:
        position = int(np.searchsorted(self.contexts, key))
        if position >= len(self.contexts) or int(self.contexts[position]) != key:
            return None
        start, end = int(self.offsets[position]), int(self.offsets[position + 1])
        candidates = self.values[start:end]
        hit = np.flatnonzero(candidates == char_id)
        if hit.size == 0:
            return None
        return float(self.logprobs[start + int(hit[0])])

    def size_bytes(self) -> int:
        return (
            self.contexts.nbytes + self.offsets.nbytes
            + self.values.nbytes + self.logprobs.nbytes
        )


class CharNGram:
    def __init__(
        self,
        order: int,
        chars: list[str],
        tables: list[_Table],
        backoff: float = 0.4,
        floor_logprob: float = -12.0,
        meta: dict | None = None,
    ):
        #: what the counts came from (running text vs bare word forms); the
        #: recognizer needs it to decide whether the model can judge spacing
        self.meta = dict(meta or {})
        self.order = int(order)
        self.chars = chars
        self.tables = tables
        self.backoff = float(backoff)
        self.floor_logprob = float(floor_logprob)
        self.index = {char: position for position, char in enumerate(chars)}
        self._bits = _bits_for(len(chars) + 1)
        self._log_backoff = float(np.log10(self.backoff))

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        order: int = 6,
        min_count: int = 3,
        backoff: float = 0.4,
        floor_logprob: float = -12.0,
        meta: dict | None = None,
    ) -> "CharNGram":
        joined = list(texts)
        text = "".join(joined)
        alphabet = sorted(set(text))
        index = {char: position + 1 for position, char in enumerate(alphabet)}  # 0 = unknown
        chars = ["\0"] + alphabet
        bits = _bits_for(len(chars))
        codes = np.fromiter(
            (index.get(char, 0) for char in text), dtype=np.int64, count=len(text)
        )
        logger.info(
            "CharNGram: %s characters, alphabet %s, order %s, min_count %s",
            len(codes), len(chars), order, min_count,
        )

        tables: list[_Table] = []
        vocabulary = len(chars)
        for current in range(1, order + 1):
            context_length = current - 1
            if context_length == 0:
                # unigrams are never pruned: they are the fallback of every
                # unseen context, and there are only a few hundred of them
                values, counts = np.unique(codes, return_counts=True)
                contexts = np.zeros(len(values), dtype=np.int64)
            else:
                # pair = packed(context) * vocabulary + next character
                packed = np.zeros(len(codes) - context_length, dtype=np.int64)
                for offset in range(context_length):
                    packed |= codes[offset : offset + packed.size] << (bits * offset)
                pairs = packed * vocabulary + codes[context_length:]
                pairs.sort()  # cheap and makes np.unique faster
                pair_values, counts = np.unique(pairs, return_counts=True)
                if min_count > 1:
                    keep = counts >= min_count
                    pair_values, counts = pair_values[keep], counts[keep]
                if len(pair_values) == 0:
                    logger.warning("CharNGram: order %s is empty after pruning", current)
                    tables.append(_Table(
                        np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64),
                        np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float16),
                    ))
                    continue
                contexts, values = np.divmod(pair_values, vocabulary)

            # pairs are unique and sorted, so equal contexts are adjacent: the
            # group heads become the CSR row keys and the rest are candidates
            starts = np.flatnonzero(np.concatenate(([True], np.diff(contexts) != 0)))
            repeats = np.diff(np.append(starts, len(contexts)))
            totals = np.add.reduceat(counts, starts)
            logprobs = np.log10(counts / np.repeat(totals, repeats)).astype(np.float16)
            offsets = np.concatenate(([0], np.cumsum(repeats))).astype(np.int64)
            tables.append(_Table(
                contexts=contexts[starts].astype(np.int64),
                offsets=offsets,
                values=values.astype(np.uint32),
                logprobs=logprobs,
            ))
            logger.info(
                "CharNGram: order %s: %s contexts, %s entries",
                current, len(contexts), len(values),
            )
        return cls(order=order, chars=chars, tables=tables,
                   backoff=backoff, floor_logprob=floor_logprob, meta=meta)

    # ------------------------------------------------------------------
    # persistence

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": np.array([FORMAT_VERSION]),
            "order": np.array([self.order]),
            "backoff": np.array([self.backoff]),
            "floor": np.array([self.floor_logprob]),
            "chars": np.array(self.chars, dtype=object),
            "meta": np.array([json.dumps(self.meta, ensure_ascii=False)], dtype=object),
        }
        for level, table in enumerate(self.tables):
            payload[f"contexts_{level}"] = table.contexts
            payload[f"offsets_{level}"] = table.offsets
            payload[f"values_{level}"] = table.values
            payload[f"logprobs_{level}"] = table.logprobs
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CharNGram":
        with np.load(path, allow_pickle=True) as data:
            if int(data["version"][0]) != FORMAT_VERSION:
                raise ValueError(f"unsupported LM format {int(data['version'][0])}")
            order = int(data["order"][0])
            tables = [
                _Table(
                    contexts=data[f"contexts_{level}"],
                    offsets=data[f"offsets_{level}"],
                    values=data[f"values_{level}"],
                    logprobs=data[f"logprobs_{level}"],
                )
                for level in range(order)  # 0 = unigrams, like self.tables
            ]
            meta = {}
            if "meta" in data:
                try:
                    meta = json.loads(str(data["meta"][0]))
                except ValueError:  # pragma: no cover - defensive
                    meta = {}
            return cls(
                order=order,
                chars=[str(char) for char in data["chars"]],
                tables=tables,
                backoff=float(data["backoff"][0]),
                floor_logprob=float(data["floor"][0]),
                meta=meta,
            )

    # ------------------------------------------------------------------
    # scoring

    @property
    def size_bytes(self) -> int:
        return sum(table.size_bytes() for table in self.tables)

    def char_id(self, char: str) -> int:
        return self.index.get(char, 0)

    def logprob(self, context: Sequence[int], char_id: int) -> float:
        """log10 P(char | context), context is the *recent* history, oldest first."""
        if char_id == 0:
            return self.floor_logprob
        longest = min(len(context), self.order - 1)
        for length in range(longest, 0, -1):
            key = self._pack(context[-length:])
            table = self.tables[length]
            value = table.lookup(key, char_id)
            if value is not None:
                return value + (longest - length) * self._log_backoff
        unigram = self.tables[0].lookup(0, char_id)
        if unigram is None:
            return self.floor_logprob + longest * self._log_backoff
        return unigram + longest * self._log_backoff

    def text_logprob(self, text: str) -> float:
        context: list[int] = []
        total = 0.0
        for char in text:
            char_id = self.char_id(char)
            total += self.logprob(context, char_id)
            context.append(char_id)
            if len(context) > self.order - 1:
                del context[0]
        return total

    # ------------------------------------------------------------------

    def _pack(self, ids: Sequence[int]) -> int:
        key = 0
        for position, char_id in enumerate(ids):
            key |= int(char_id) << (self._bits * position)
        return key

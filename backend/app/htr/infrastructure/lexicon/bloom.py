"""Bloom filter over normalized word forms.

A Russian word-form list has ~1.5 million entries; a Python ``set`` of that size
costs a couple of hundred megabytes, while the same list in a Bloom filter fits
in ~2 MB and answers in constant time. The only error mode is a *false
positive* (an unknown word looks known), which for this feature means one less
red flag — acceptable, and measured by ``scripts/calibrate_htr_lexicon.py``.

File format: ``b"HTRLEX1\\n"`` + 4-byte big-endian header length + JSON header +
raw bit array.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MAGIC = b"HTRLEX1\n"
_DIGEST_SIZE = 16


def _bit_count_for(item_count: int, false_positive_rate: float) -> int:
    if item_count <= 0:
        return 64
    bits = -item_count * math.log(false_positive_rate) / (math.log(2) ** 2)
    return max(64, int(math.ceil(bits)))


def hash_count_for(bit_count: int, item_count: int) -> int:
    if item_count <= 0:
        return 1
    hashes = round(bit_count / item_count * math.log(2))
    return max(1, min(16, hashes))


class BloomFilter:
    def __init__(self, bits: bytearray, hash_count: int, item_count: int = 0, meta: dict | None = None):
        self.bits = bits
        self.hash_count = max(1, int(hash_count))
        self.item_count = int(item_count)
        self.meta = dict(meta or {})

    # -- construction --------------------------------------------------

    @classmethod
    def build(
        cls,
        capacity: int,
        false_positive_rate: float = 0.01,
        meta: dict | None = None,
    ) -> "BloomFilter":
        bit_count = _bit_count_for(capacity, false_positive_rate)
        return cls(
            bytearray((bit_count + 7) // 8),
            hash_count_for(bit_count, capacity),
            item_count=0,
            meta={**(meta or {}), "bit_count": bit_count, "false_positive_rate": false_positive_rate},
        )

    @property
    def bit_count(self) -> int:
        return len(self.bits) * 8

    def _positions(self, word: str):
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=_DIGEST_SIZE).digest()
        first = int.from_bytes(digest[:8], "little")
        second = int.from_bytes(digest[8:], "little") | 1
        mask = self.bit_count - 1 if self.bit_count & (self.bit_count - 1) == 0 else None
        for index in range(self.hash_count):
            value = first + index * second
            yield value & mask if mask is not None else value % self.bit_count

    # -- use -----------------------------------------------------------

    def add(self, word: str) -> None:
        for position in self._positions(word):
            self.bits[position >> 3] |= 1 << (position & 7)
        self.item_count += 1

    def extend(self, words: Iterable[str]) -> int:
        added = 0
        for word in words:
            self.add(word)
            added += 1
        return added

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        return all(
            self.bits[position >> 3] & (1 << (position & 7))
            for position in self._positions(word)
        )

    def __len__(self) -> int:
        return self.item_count

    # -- persistence ---------------------------------------------------

    def to_bytes(self) -> bytes:
        header = json.dumps(
            {
                "hash_count": self.hash_count,
                "item_count": self.item_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **self.meta,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return MAGIC + len(header).to_bytes(4, "big") + header + bytes(self.bits)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_bytes())
        return target

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        if not data.startswith(MAGIC):
            raise ValueError("not a lexicon artifact (bad magic)")
        offset = len(MAGIC)
        header_length = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        header = json.loads(data[offset : offset + header_length].decode("utf-8"))
        offset += header_length
        bits = bytearray(data[offset:])
        if not bits:
            raise ValueError("lexicon artifact has an empty bit array")
        meta = {k: v for k, v in header.items() if k not in ("hash_count", "item_count")}
        return cls(bits, int(header["hash_count"]), int(header.get("item_count", 0)), meta)

    @classmethod
    def load(cls, path: str | Path) -> "BloomFilter":
        return cls.from_bytes(Path(path).read_bytes())

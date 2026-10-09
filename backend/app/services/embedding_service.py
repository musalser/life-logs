"""Эмбеддинги для дедупликации знаний и RAG (google/embeddinggemma-2, 768d).

The weights are loaded **in-process** by sentence-transformers instead of being
served by Ollama:

* the encoder is a plain local model, so an embedding never queues behind a
  14b chat generation (which can hold the Ollama worker for minutes),
* the archive and its model live together under ``htr_storage/embedding_models``
  (gitignored) instead of inside a container runtime's model store,
* nothing here needs a network once the weights are downloaded.

What this service adds on top of a bare ``SentenceTransformer``:

* lazy loading behind a thread lock (the first call pays for the weights),
* batch encoding off the event loop, so a long encode cannot block the API,
* task instruction prefixes (the model card's prompts): symmetric
  similarity for deduplication, query/document for RAG,
* the weights load in bfloat16 (the checkpoint's own dtype). The card forbids
  float16 — its activation range overflows it and the model returns silently
  degraded vectors instead of an error; bfloat16 was measured identical to
  float32 on our pairs, so nothing is cast,
* a bounded LRU cache keyed by model + prompt + text (a re-extracted page asks
  for the same titles over and over),
* a dimension check: the pgvector column is fixed at ``embedding_dim``, so a
  model mismatch must fail loudly here, not at INSERT time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any, Sequence

from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot produce usable vectors."""


class EmbeddingService:
    def __init__(
        self,
        encoder: Any | None = None,
        model: str | None = None,
        model_dir: str | None = None,
        device: str | None = None,
        prompt: str | None = None,
        dim: int | None = None,
        batch_size: int | None = None,
        cache_size: int | None = None,
    ) -> None:
        #: the encoder is injectable so tests never download 1.5 GB of weights
        self._encoder = encoder
        self.model_name = model or settings.embedding_model
        self.model_dir = settings.embedding_model_dir if model_dir is None else model_dir
        self.device = device or settings.embedding_device
        self.default_prompt = settings.embedding_default_prompt if prompt is None else prompt
        self.dim = dim or settings.embedding_dim
        self.batch_size = batch_size or settings.embedding_batch_size
        self.cache_size = settings.embedding_cache_size if cache_size is None else cache_size
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_texts(
        self,
        texts: Sequence[str],
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embeds every text, reusing the cache.

        ``prompt_name`` is the model-card task prefix. ``None`` means "use the
        configured default" (symmetric similarity for deduplication), an empty
        string means "no prefix at all"; RAG passes ``'SearchQuery'`` /
        ``'Document'`` explicitly.
        """
        if not texts:
            return []

        prompt = self.default_prompt if prompt_name is None else (prompt_name or None)
        cleaned = [str(text).strip() for text in texts]
        for text in cleaned:
            if not text:
                raise EmbeddingError("cannot embed an empty string")

        results: list[list[float] | None] = [None] * len(cleaned)
        pending: list[tuple[int, str]] = []
        for index, text in enumerate(cleaned):
            cached = self._cache_get(text, prompt)
            if cached is None:
                pending.append((index, text))
            else:
                results[index] = cached

        if pending:
            vectors = await asyncio.to_thread(
                self._encode_sync, [text for _, text in pending], prompt
            )
            if len(vectors) != len(pending):
                raise EmbeddingError(
                    f"embedding model {self.model_name!r} returned {len(vectors)} "
                    f"vectors for {len(pending)} inputs"
                )
            for (index, text), vector in zip(pending, vectors):
                self._validate(vector)
                results[index] = vector
                self._cache_put(text, prompt, vector)

        if any(vector is None for vector in results):
            raise EmbeddingError("embedding backend returned fewer vectors than inputs")
        return [vector for vector in results if vector is not None]

    async def embed_text(self, text: str, prompt_name: str | None = None) -> list[float]:
        return (await self.embed_texts([text], prompt_name=prompt_name))[0]

    def embed_texts_sync(
        self, texts: Sequence[str], prompt_name: str | None = None
    ) -> list[list[float]]:
        """Blocking wrapper for Celery workers (no running event loop there)."""
        return asyncio.run(self.embed_texts(texts, prompt_name=prompt_name))

    def embed_text_sync(self, text: str, prompt_name: str | None = None) -> list[float]:
        return self.embed_texts_sync([text], prompt_name=prompt_name)[0]

    async def warmup(self) -> None:
        """Loads the weights at startup when ``embedding_warmup_enabled`` is set."""
        if not settings.embedding_warmup_enabled:
            return
        try:
            dimension = await self.probe()
            logger.info("Embedding model %s is ready (%s dims)", self.model_name, dimension)
        except EmbeddingError as exc:
            logger.warning("Embedding warmup failed: %s", exc)

    async def probe(self) -> int:
        """Embeds a probe string and returns the vector length.

        Answers "is the configured model really loadable and does it match
        embedding_dim?" — used by readiness checks and after a model change.
        """
        vector = await self.embed_text("проверка эмбеддингов", prompt_name="")
        return len(vector)

    @staticmethod
    def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        """Cosine similarity of two vectors, without pulling numpy in."""
        if not left or not right:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        norm_left = sum(a * a for a in left) ** 0.5
        norm_right = sum(b * b for b in right) ** 0.5
        if norm_left == 0.0 or norm_right == 0.0:
            return 0.0
        return dot / (norm_left * norm_right)

    # ------------------------------------------------------------------
    # Encoder loading
    # ------------------------------------------------------------------

    @property
    def encoder(self) -> Any:
        """The loaded SentenceTransformer (downloads the weights on first use)."""
        if self._encoder is None:
            with self._load_lock:
                if self._encoder is None:
                    self._encoder = self._load_encoder()
        return self._encoder

    def _load_encoder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EmbeddingError(
                "sentence-transformers is not installed: run "
                "`uv pip install 'sentence-transformers>=6.1.0'`"
            ) from exc

        device = self._resolve_device()
        #: the checkpoint is a multimodal one (text + vision + audio); for an
        #: archive of Russian prose only the text tower is ever used, and the
        #: model card documents dropping the other two. Measured: identical
        #: vectors, 1.5 GB -> ~0.5 GB of weights per process.
        config_kwargs = (
            {"vision_config": None, "audio_config": None}
            if settings.embedding_text_only
            else {}
        )
        logger.info(
            "Loading embedding model %s on %s (cache: %s, text_only=%s)",
            self.model_name,
            device,
            self.model_dir or "hugging face default",
            settings.embedding_text_only,
        )
        try:
            model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=self.model_dir or None,
                config_kwargs=config_kwargs,
            )
        except Exception as exc:
            raise EmbeddingError(f"cannot load embedding model {self.model_name!r}: {exc}") from exc
        logger.info(
            "Embedding model loaded: %sM parameters in %s",
            sum(parameter.numel() for parameter in model.parameters()) // 1_000_000,
            next(model.parameters()).dtype,
        )

        # the checkpoint declares an absurd max_seq_length (1e33); the model card
        # documents 8192 tokens, and a real cap keeps a long chunk from asking
        # for an enormous attention buffer
        if settings.embedding_max_seq_length > 0:
            model.max_seq_length = min(
                int(getattr(model, "max_seq_length", settings.embedding_max_seq_length)),
                settings.embedding_max_seq_length,
            )
        return model

    def _resolve_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:  # pragma: no cover - torch ships with the project
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _encode_sync(self, texts: list[str], prompt_name: str | None) -> list[list[float]]:
        encoder = self.encoder
        #: one encoder, one GPU: serialise the forward passes of concurrent
        #: requests instead of letting them fight for memory
        with self._encode_lock:
            try:
                vectors = encoder.encode(
                    list(texts),
                    prompt_name=prompt_name,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"embedding model {self.model_name!r} failed to encode: {exc}"
                ) from exc
        return [[float(value) for value in vector] for vector in vectors]

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _validate(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dim:
            raise EmbeddingError(
                f"embedding model {self.model_name!r} returned {len(vector)}-dimensional "
                f"vectors, but embedding_dim is {self.dim}: set EMBEDDING_DIM to match the model"
            )

    def _cache_key(self, text: str, prompt_name: str | None) -> str:
        # the model and the task prefix are part of the key: switching either
        # must not serve vectors produced under the old one
        return f"{self.model_name}\x00{prompt_name or ''}\x00{text}"

    def _cache_get(self, text: str, prompt_name: str | None) -> list[float] | None:
        key = self._cache_key(text, prompt_name)
        vector = self._cache.get(key)
        if vector is not None:
            self._cache.move_to_end(key)
        return vector

    def _cache_put(self, text: str, prompt_name: str | None, vector: list[float]) -> None:
        if self.cache_size <= 0:
            return
        key = self._cache_key(text, prompt_name)
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)


_embedding_service: EmbeddingService | None = None


def get_embedding_service(encoder: Any | None = None) -> EmbeddingService:
    """Process-wide singleton (its cache and its loaded weights are the point).

    The first caller may inject an encoder (tests); a later call returns the
    service already built.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(encoder=encoder)
    return _embedding_service

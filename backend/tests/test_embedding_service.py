"""Юнит-тесты сервиса эмбеддингов: батчи, кэш, промпты, размерность.

Ни сеть, ни веса не нужны: энкодер подменяется фейком, поэтому тесты проверяют
именно ту логику, которую сервис добавляет к sentence-transformers.
"""

import asyncio

import pytest

from app.services import embedding_service as embedding_module
from app.services.embedding_service import EmbeddingError, EmbeddingService


class FakeEncoder:
    """Повторяет ту часть API SentenceTransformer, которой пользуется сервис."""

    def __init__(self, dim: int = 4, fail_times: int = 0):
        self.dim = dim
        self.fail_times = fail_times
        self.calls: list[dict] = []

    def encode(self, texts, prompt_name=None, batch_size=None, normalize_embeddings=True, show_progress_bar=False):
        self.calls.append({"texts": list(texts), "prompt_name": prompt_name})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("model is not loaded")
        return [[float(len(text) + index) for index in range(self.dim)] for text in texts]


def run(coro):
    return asyncio.run(coro)


def make_service(encoder, **kwargs):
    """Сервис под фейковый энкодер: размерность берётся у фейка, не из настроек."""
    return EmbeddingService(encoder=encoder, dim=encoder.dim, **kwargs)


def test_embed_texts_uses_the_default_prompt():
    encoder = FakeEncoder()
    service = make_service(encoder)

    run(service.embed_texts(["цель"]))

    assert encoder.calls[0]["prompt_name"] == service.default_prompt


def test_embed_texts_forwards_an_explicit_prompt():
    encoder = FakeEncoder()
    service = make_service(encoder)

    run(service.embed_texts(["запрос"], prompt_name="SearchQuery"))

    assert encoder.calls[0]["prompt_name"] == "SearchQuery"


def test_empty_prompt_name_means_no_prefix():
    encoder = FakeEncoder()
    service = make_service(encoder)

    run(service.embed_texts(["запрос"], prompt_name=""))

    assert encoder.calls[0]["prompt_name"] is None


def test_embed_texts_reuses_cache():
    encoder = FakeEncoder()
    service = make_service(encoder)

    run(service.embed_texts(["цель", "привычка"]))
    run(service.embed_texts(["цель", "привычка"]))

    assert len(encoder.calls) == 1  # второй раз всё пришло из кэша


def test_cache_is_keyed_by_prompt():
    encoder = FakeEncoder()
    service = make_service(encoder)

    run(service.embed_texts(["цель"], prompt_name="Clustering"))
    run(service.embed_texts(["цель"], prompt_name="SearchQuery"))

    assert len(encoder.calls) == 2  # разные префиксы — разные векторы


def test_cache_is_bounded():
    encoder = FakeEncoder()
    service = make_service(encoder, cache_size=2)

    run(service.embed_texts(["a", "b", "c"]))

    assert len(service._cache) == 2


def test_misses_are_encoded_in_one_call():
    encoder = FakeEncoder()
    service = make_service(encoder)

    run(service.embed_texts(["a", "b", "c"]))

    assert [len(call["texts"]) for call in encoder.calls] == [3]


def test_dimension_mismatch_is_loud():
    service = EmbeddingService(encoder=FakeEncoder(dim=8), dim=4)

    with pytest.raises(EmbeddingError, match="dimensional"):
        run(service.embed_texts(["цель"]))


def test_empty_text_is_rejected():
    service = make_service(FakeEncoder())

    with pytest.raises(EmbeddingError, match="empty"):
        run(service.embed_texts(["  "]))


def test_encoder_failure_is_wrapped():
    service = make_service(FakeEncoder(fail_times=1))

    with pytest.raises(EmbeddingError, match="failed to encode"):
        run(service.embed_texts(["цель"]))


def test_cosine_similarity():
    assert EmbeddingService.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert EmbeddingService.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert EmbeddingService.cosine_similarity([], [1.0]) == 0.0
    assert EmbeddingService.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_sync_wrapper():
    service = make_service(FakeEncoder())

    assert len(service.embed_text_sync("цель")) == 4
    assert len(service.embed_texts_sync(["цель", "привычка"])) == 2


def test_probe_reports_the_dimension():
    assert run(make_service(FakeEncoder(dim=3)).probe()) == 3


def test_encoder_is_loaded_once():
    encoder = FakeEncoder()
    service = make_service(encoder)

    assert service.encoder is encoder
    assert service.encoder is encoder


def test_singleton_keeps_the_first_encoder():
    first = FakeEncoder()
    singleton = embedding_module.get_embedding_service(encoder=first)
    assert singleton.encoder is first
    assert embedding_module.get_embedding_service(encoder=FakeEncoder()) is singleton
    # не течёт в другие тесты: синглтон процесс-глобальный
    embedding_module._embedding_service = None

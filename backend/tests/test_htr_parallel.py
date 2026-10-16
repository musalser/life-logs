"""The worker processes give exactly what the in-process path gives."""
from __future__ import annotations

import numpy as np
import pytest

from app.htr.infrastructure.kraken import parallel
from app.htr.infrastructure.kraken.beam import BeamSearchConfig, PrefixBeamSearch

pytest.importorskip("kraken")


@pytest.fixture()
def two_workers():
    parallel.configure(2)
    yield
    parallel.shutdown()
    parallel.configure(1)


def _codec(charset: str):
    from kraken.lib.codec import PytorchCodec

    return PytorchCodec(list(charset))


def test_decoding_in_workers_matches_in_process(two_workers):
    codec = _codec("акорв ")
    rng = np.random.default_rng(7)
    config = BeamSearchConfig(beam_width=8, top_k=4, min_prob=1e-3)
    search = PrefixBeamSearch(codec=codec, config=config)
    matrices = [rng.random((len(codec.l2c_single) + 1, 30)).astype(np.float32) for _ in range(6)]
    frames = [search.frame_candidates(search.probabilities(m)) for m in matrices]
    spec = parallel.BeamSpec(graphemes=list(search._graphemes), lm_path=None, config=config)

    texts = parallel.decode_lines(spec, frames, n=1)
    nbest = parallel.decode_lines(spec, frames, n=3)

    assert texts == [search.decode(m) for m in matrices]
    assert [[c.text for c in line] for line in nbest] == [
        [c.text for c in search.decode_nbest(m, n=3)] for m in matrices
    ]


def test_outlines_in_workers_match_in_process(two_workers):
    from scipy.ndimage import gaussian_filter
    from skimage.filters import sobel

    image = np.full((200, 400), 255, dtype=np.uint8)
    baselines = []
    for row in range(4):
        y = 40 + row * 40
        image[y - 12:y, 30:370] = 40
        baselines.append([[30, y], [370, y]])
    im_feats = gaussian_filter(sobel(image), 0.5)
    targets = [
        (line, [other for other in baselines if other is not line]) for line in baselines
    ]

    local = parallel._polygonize_local(im_feats, targets, False, False)
    pooled = parallel.polygonize(im_feats, targets)

    assert len(pooled) == len(local) == 4
    for a, b in zip(local, pooled):
        assert (a is None and b is None) or np.array_equal(np.asarray(a), np.asarray(b))


def test_a_single_worker_never_starts_a_pool():
    parallel.configure(1)
    assert parallel._pool() is None

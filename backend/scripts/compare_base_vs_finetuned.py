"""Сравнение базовой модели с beam+rescoring против дообученной v13.

Отвечает на вопрос «стоит ли дообученная модель больше, чем улучшения декодера
поверх базовой». Эталон пользователя недоступен, если страницы не вычитаны,
поэтому метрики выбраны так, чтобы он не требовался:

* **OOV-доля** (слова вне общего словаря) — прокси качества текста: чем меньше
  слов вне словаря, тем лучше чтение. Считается по общему словарю; лексикон
  автора добавляется только при живой БД, поэтому абсолютные значения завышены,
  но сравнение честное — линейка одна для обоих конвейеров;
* **влияние char LM** (greedy -> beam) и **rescoring** (beam -> +словесная LM);
* **расхождение конвейеров** (CER/WER одного против другого) — насколько
  изменится текст, если переключиться.

Матрицы кэшированы (`htr_storage/lm/matrices/page_<id>[_v13].npz`), поэтому
распознавание не запускается и результат детерминированный.

Usage (from the ``backend`` directory)::

    .venv/bin/python scripts/compare_base_vs_finetuned.py


Матрицы уже кэшированы для базовой модели и для v13, поэтому распознавание не
нужно. Метрики выбраны так, чтобы не требовался эталон пользователя:
  * OOV-доля (слова не из общего словаря) — прокси качества текста;
  * влияние LM (greedy -> beam) и rescoring (beam -> rescored);
  * расхождение между конвейерами (CER/WER A против B).
"""
import sys
sys.path.insert(0, "/home/musalser/projects/life-logs/backend")
import unicodedata
from pathlib import Path
import numpy as np
import kenlm

from app.config import settings
from app.htr.factory import build_beam_config, build_recognizer
from app.htr.application.metrics import MetricsEvaluator
from app.htr.domain.text import iter_words
from app.htr.infrastructure.kraken.beam import PrefixBeamSearch
from app.htr.infrastructure.lm.char_ngram import CharNGram
from app.htr.infrastructure.lm.word_rescorer import (
    KenLMSentenceScorer, RescoreConfig, WordRescorer,
)
from app.htr.infrastructure.lexicon import LayeredLexiconChecker, load_file_lexicon

PAGES = [23, 24]
ALPHA = settings.htr_beam_alpha
RESCORE_W = 1.0   # подобранное значение (LOO) на строгом эталоне

lexicon = load_file_lexicon(settings.htr_lexicon_path)
known = LayeredLexiconChecker(base=lexicon)
print(f"словарь: доступен={lexicon.is_available}, alpha={ALPHA}, w={RESCORE_W}", flush=True)

word_model = kenlm.Model(str(Path(settings.htr_storage_dir) / "lm" / "word_ru.binary"))
rescorer = WordRescorer(
    KenLMSentenceScorer(word_model),
    RescoreConfig(weight=RESCORE_W, guard=False),
    known_word=lambda word: known.is_known(word),
)

def oov_rate(text: str) -> tuple[int, int]:
    words = [w for w in iter_words(unicodedata.normalize("NFC", text))]
    unknown = [w for w in words if not known.is_known(w)]
    return len(unknown), len(words)

def load_cache(name: str):
    with np.load(Path(settings.htr_storage_dir) / "lm" / "matrices" / name, allow_pickle=True) as data:
        return [np.asarray(m, dtype=np.float32) for m in data["matrices"]], [str(g) for g in data["greedy"]]

def build_search(page_id: int):
    lm = CharNGram.load(f"htr_storage/lm/folds/chrono_leipzig_{page_id}_char_lm.npz")
    config = build_beam_config()
    config.alpha = ALPHA
    return lm, config

evaluator = MetricsEvaluator()
summary = {}

for label, tag in (("base", ""), ("v13", "_v13")):
    totals = {"greedy_oov": 0, "beam_oov": 0, "resc_oov": 0, "words": 0}
    lm_effect, rescore_effect = [], []
    texts = {}
    for page_id in PAGES:
        lm, config = build_search(page_id)
        # кодек у обеих моделей одинаковый (1622 класса), но матрицы сняты
        # разными сетями, поэтому берём именно ту, что дала этот кэш
        net, _ = build_recognizer()._load_model(
            Path(settings.htr_storage_dir) / "models" / ("author_1/v13/model.safetensors" if tag else "default/ppocrv6_medium.safetensors")
        )
        search = PrefixBeamSearch(codec=net.codec, lm=lm, config=config)
        matrices, greedy = load_cache(f"page_{page_id}{tag}.npz")
        page_texts = []
        for matrix, greedy_text in zip(matrices, greedy):
            candidates = search.decode_nbest(matrix, n=10)
            beam_text = candidates[0].text
            rescored = rescorer.choose(candidates).text
            page_texts.append((greedy_text, beam_text, rescored))
            lm_effect.append((greedy_text, beam_text))
            rescore_effect.append((beam_text, rescored))
            for text, key in ((greedy_text, "greedy_oov"), (beam_text, "beam_oov"), (rescored, "resc_oov")):
                unknown, total = oov_rate(text)
                totals[key] += unknown
                if key == "beam_oov":
                    totals["words"] += total
        texts[page_id] = page_texts
        print(f"  {label} стр.{page_id}: строк {len(page_texts)}", flush=True)
    summary[label] = {
        "totals": totals,
        "lm_effect": evaluator.evaluate_pairs(lm_effect),
        "rescore_effect": evaluator.evaluate_pairs(rescore_effect),
        "texts": texts,
    }

print("\n=== метрики без эталона (страницы 23, 24) ===", flush=True)
for label in ("base", "v13"):
    data = summary[label]
    t = data["totals"]
    print(f"\n{label}:")
    print(f"  OOV: greedy {t['greedy_oov']}/{t['words']} ({t['greedy_oov']/max(1,t['words'])*100:.1f} %)"
          f" | beam {t['beam_oov']}/{t['words']} ({t['beam_oov']/max(1,t['words'])*100:.1f} %)"
          f" | +rescoring {t['resc_oov']} ({t['resc_oov']/max(1,t['words'])*100:.1f} %)")
    print(f"  влияние char LM (greedy -> beam): CER {data['lm_effect']['cer']:.4f} WER {data['lm_effect']['wer']:.4f}")
    print(f"  влияние rescoring (beam -> +LM): CER {data['rescore_effect']['cer']:.4f} WER {data['rescore_effect']['wer']:.4f}")

print("\n=== расхождение конвейеров (A = базовая+beam+rescoring, B = v13+beam) ===", flush=True)
pairs = []
for page_id in PAGES:
    for (_g, _b, a_text), (_g2, b_text, _r) in zip(summary["base"]["texts"][page_id], summary["v13"]["texts"][page_id]):
        pairs.append((b_text, a_text))
gap = evaluator.evaluate_pairs(pairs)
print(f"  A против B: CER {gap['cer']:.4f} WER {gap['wer']:.4f} (строк {gap['lines']})")

print("\n=== примеры расхождений (первые 6) ===", flush=True)
shown = 0
for page_id in PAGES:
    for (_g, _b, a_text), (_g2, b_text, _r) in zip(summary["base"]["texts"][page_id], summary["v13"]["texts"][page_id]):
        if a_text != b_text and shown < 6:
            print(f"  стр.{page_id}")
            print(f"    A (base+beam+rescoring): {a_text[:74]!r}")
            print(f"    B (v13+beam):            {b_text[:74]!r}")
            shown += 1

# HTR module (handwriting recognition & per-author training)

Recognition pipeline: page image → binarization → line segmentation → CTC
recognition → editable text. Training fine-tunes a per-author model on the
confirmed lines of that author.

Dependencies point inward: `API → application → domain ← infrastructure`.
Kraken is imported **only** inside `infrastructure/kraken/` and only lazily, so
the application runs without the optional HTR backend (recognition then answers
`422` with an actionable message instead of crashing).

## Line segmentation and polygons

Handwritten notebook photos have curved, slanted lines, so axis-aligned boxes
cut through the text and split one line into two. The recognizer therefore uses
kraken's bundled **neural bLLA model**
(`kraken.tasks.SegmentationTaskModel.load_model`), which returns *baseline
polygons*: a bounding polygon per line plus per-word outlines. On the
development page (3472×4640, leaves curling up) this yields **36 lines instead
of 45**, and every line/word carries a polygon that follows the handwriting.

* `htr_segmentation_engine = "neural"` (default) — bLLA baselines;
  `"classical"` keeps the projection-profile segmenter (`kraken.pageseg`), which
  only produces straight boxes. The neural path falls back to the classical one
  with a warning if the model cannot be loaded (e.g. no `coremltools`).
* Loading the bLLA weights needs `coremltools` (CoreML container format), which
  has Linux/WSL wheels but none for Windows.
* Polygons are stored per line and per word (`htr_lines.polygon`,
  `htr_words.polygon`, JSON `[[x, y], …]`); the axis-aligned `bbox` remains as a
  cheap envelope, and the UI falls back to it when a page has no polygon (older
  recognitions).
* `POST /htr/pages/{id}/recognize?force=true` re-segments a page that is already
  fixed: an `EDITING` page loses its corrections, a `CONFIRMED` page loses its
  confirmation too (it drops back to `RECOGNIZED`; `confirmed_at` and the
  prediction CER/WER are cleared). The UI asks for confirmation first.

### One line split into two records

bLLA occasionally detaches the first word of a line into its own record, so the
transcript gets a one-word row followed by the rest of the line. Before
recognition the adapter glues such pieces back together
(`merge_collinear_lines`), because merging afterwards would only hide the split.

The decision is geometric and **direction aware**: the vector between two pieces
is projected onto the local baseline direction, and a merge requires

* a small *perpendicular* component (the pieces lie on the same baseline) and
* an *along-baseline* gap within about one line pitch (a word space, not a
  column gap) and
* roughly parallel local slopes.

The tolerance scale is the **line pitch** estimated from the page
(`estimate_pitch`: span of the baseline midpoints / gaps), *not* the glyph
height — the outline of a wavy line can be taller than the line spacing, which
made an earlier height-based version glue real consecutive lines together
(23 lines instead of 32 on the development page). Detached words are also
handled when they end up non-adjacent in reading order: pieces are grouped with
union-find and the merged record keeps the position of its earliest piece.

Measured on the development page: **36 → 32 lines**, `сейчас`/`успевал` are back
inside their lines, word count unchanged (200 → 199). Disable the step with
`htr_segmentation_merge_lines = false` if a page ever merges too eagerly.

## Model proposals: the recognition is never overwritten

After recognition the page can be sent through the Ollama instance the chat
uses (`gemma3:12b`, see the settings below). The answer is **not** written into
the transcription. Three kinds of text are kept apart:

| column | who writes it | role |
| --- | --- | --- |
| `predicted_text` | kraken | the recognition itself; nothing modifies it, ever |
| `suggested_text` + `suggested_by` | the language model | a proposal, shown as a diff and waiting for a decision |
| `corrected_text` + `corrected_by` | the user | the only ground truth: hand edits and accepted proposals |

So a fine-tune can never learn from unreviewed machine output, and the CER/WER
of a confirmed page keeps measuring the recognizer, not the language model.

The prompt is strict: fix only mechanical errors, never touch the author's
style, orthography, punctuation or phrasing, keep a word the model is not sure
about, answer with the corrected line and nothing else.

**What the model gets besides the text** (built from what the project already
stores — this is "Phase 0"):

| part | source |
| --- | --- |
| author lexicon | frequent words of the **confirmed** pages (the user's transcription, never the raw prediction) |
| character confusions | `prediction ↔ correction` alignments, e.g. `н→п (14)`, `ш→н (6)` |
| few-shot examples | real `было: … / стало: …` line pairs of this author |
| domain vocabulary | entities, event/goal/habit titles from the knowledge base (proper nouns a generic LM cannot guess) |
| neighbouring lines | the ±`htr_correction_context_lines` lines around the target, as context only |

### Reviewing a proposal

The proposal is stored as a whole line, but a human reviews *changes*, so the
word-level diff is computed on every read (`application/suggestions.py`) and
each replaced/inserted word carries a dictionary verdict:

* **green** — the new word exists in the general dictionary or in the author's
  own vocabulary. This is where `htr_lexicon_*` (see the next section) pays off
  twice: the same dictionary that marks suspicious words *also* validates what
  the model wants to write;
* **violet** — nobody has ever seen that word. This is where a language model
  invents things (`и.` → `г.` was a real case on the development pages);
* **grey** — nothing to check against (no dictionary installed at all).

**Every chip is a button**: clicking one applies exactly that word and leaves
the rest of the proposal in place, for the frequent case where the model is
right about «очен → очень» but wrong about the «Судиславль» it dropped. The
diff is recomputed against the new text, so the remaining chips shrink as the
line converges; when nothing is left, the proposal disappears. «Принять всё»
takes the whole line (the only way for a punctuation-only proposal), «Скрыть»
drops the proposal without touching the text.

Applying a single change splices the original text at the character offsets of
that word (`application/suggestions.py`), so the author's punctuation and
spacing survive: the period the model moved stays where the author put it.
A word the segmenter glued or split («на зывалось» → «называлось») is recognised
as a single change rather than a replacement plus a deletion, so one click fixes
it and the dictionary can verify it.

A line is *verified* only when every change is green, and
«Принять проверенные (N)» accepts exactly those lines. Everything else — a
violet change, a deleted word, or a missing dictionary — is left for the user
to read, one line at a time. Pure punctuation proposals carry no word change,
so nothing about them can be invented.

Other rules:

* lines the user already wrote themselves are skipped: a proposal about the
  discarded raw reading would only be noise;
* a hand edit drops the pending proposal of that line, and confirming a page
  clears the proposals that are still open;
* `words_stale` is set when a proposal is accepted with a different word count,
  i.e. when the stored word boxes really stop matching the text.

**Best-effort by design**: if Ollama is unavailable, the model is still loading
or the answer looks unusable (empty, multi-line, or 3× shorter/longer than the
input), that line simply gets no proposal and a warning is logged. Recognition
never fails because of the LLM step.

| setting | default | meaning |
| --- | --- | --- |
| `htr_correction_enabled` | `true` | expose the suggester (`POST /suggestions`, UI button) |
| `htr_correction_auto` | `false` | also ask for proposals right after recognition |
| `htr_correction_model` | `gemma3:12b` | Ollama model (same instance as the chat) |
| `htr_correction_timeout_s` | `90` | per-line *generation* timeout (a cold 12b model needs ~1 minute) |
| `htr_correction_connect_timeout_s` | `5` | connect timeout, kept separate: a stopped Ollama behind the Windows firewall drops packets, and one shared budget made every dead attempt cost the full 90 s |
| `htr_correction_context_lines` | `2` | neighbouring lines given as context |
| `htr_correction_max_lexicon` / `_max_vocabulary` | `200` / `200` | prompt size caps |
| `htr_correction_max_confusions` / `_max_examples` | `25` / `5` | prompt size caps |

## Dictionary check of the recognized words (OOV)

The confidence score catches a hesitant model, not a **confident mistake**: the
recognizer reports 95 % for a word it read wrong. A dictionary catches those,
together with proper nouns and dialect words that are in no word list at all.
Every word therefore carries `in_lexicon`, and the UI marks the misses:

* **violet** is reserved for "not in the dictionary" (fill of the word polygon,
  a line badge, a chip under the line, a per-page counter in the sidebar);
* **amber/red** stay the confidence scale (polygon outline), so both signals are
  visible on the same word instead of competing for one colour.

**What counts as known**

1. the general Russian dictionary (`htr_storage/lexicon/ru_lexicon.bloom`);
2. the words of the author's **confirmed** pages — their own transcription, i.e.
   names and dialect words the user really writes;
3. entities, event/goal/habit titles of the knowledge base
   (`VocabularyProvider`);
4. one-letter words, numbers and dates (`1917г`), anything with a digit, and a
   hyphenated word whose parts are all known (`кто-то`).

Old orthography is handled by *adding lookup candidates* rather than rewriting
the text: `normalize_word`/`word_variants` (`domain/text.py`) map `ѣ і ѳ ѵ`, a
word-final `ъ` and the old endings (`-аго → -ого`, `-ыя → -ые`, `-ія → -ие`,
`ея → её`). A word is accepted if **any** candidate is in the dictionary, so an
aggressive rule can only ever turn a red flag into a green one.

**The artifact** is a Bloom filter built once by an offline script — ~2.9 MB for
2.4 M word forms and ~2 MB of RAM at runtime, instead of the few hundred
megabytes a Python `set` of the same list would need. Its only error mode is a
false positive (an unknown word looks known), which costs one less flag.

```bash
.venv/bin/python scripts/build_htr_lexicon.py            # download + build
.venv/bin/python scripts/build_htr_lexicon.py --no-default-sources --source <file>
.venv/bin/python scripts/calibrate_htr_lexicon.py --author-id 1
```

The default sources are the MIT-licensed
[danakt/russian-words](https://github.com/danakt/russian-words) lists (word
forms + surnames, windows-1251, downloaded into
`htr_storage/lexicon/downloads/`, git-ignored). `--include-database` also bakes
in the confirmed words and knowledge terms, but at runtime those are added
**live** anyway, so rebuilding is only needed for a new general word list.

**When the artifact is missing** the check is simply unavailable:
`lexicon_available = false`, `in_lexicon = null`, no word is marked and the UI
says «словарь не установлен». A page is never painted violet wholesale.

**Calibration matters**: on the user's own confirmed text every OOV word is a
potential false alarm, so `calibrate_htr_lexicon.py` reports the OOV rate per
page and overall (target ≤ ~3 %) plus the most frequent misses. On the two
development pages (unconfirmed, raw predictions) the checker marks 14.6 % of the
words and almost all of them are real misreadings (`тогдо`, `Жизь`,
`организоцни`, `сиравился`) — that is the intended signal; the rate on confirmed
text is what tells whether the dictionary fits the corpus.

| setting | default | meaning |
| --- | --- | --- |
| `htr_lexicon_enabled` | `true` | run the dictionary check on page reads |
| `htr_lexicon_path` | `htr_storage/lexicon/ru_lexicon.bloom` | the artifact |

Read path: `application/lexicon.py` (`LexiconAnnotator`) fills the fields on
every page read and on the sidebar rows; `infrastructure/lexicon/` holds the
Bloom filter, the per-author corpus (confirmed words + page transcriptions, one
cached snapshot invalidated by `page_repository` on every write) and the layered
checker. Nothing is stored in the database — the annotation is recomputed, so a
new dictionary or a newly confirmed page takes effect immediately.

## Beam-search decoding with a character language model

kraken decodes CTC greedily: at every time step it takes the most probable
character on its own. A character that is locally plausible but globally wrong
is never reconsidered. Prefix beam search keeps several hypotheses alive and
scores whole prefixes with a language model, which fixes exactly the errors the
dictionary check can only *flag*, never repair.

`infrastructure/kraken/beam.py` implements CTC prefix beam search over the same
matrix the greedy decoder uses. `torchaudio` and `pyctcdecode` are unavailable
for this Python, and `kenlm` does not build on 3.13, so both the search and the
n-gram model are ours:

* the matrix is obtained with `RecognitionInferenceConfig(return_logits=True)`;
  `record.logits` is `(classes, time)`, **already softmaxed**, `blank = 0`
  (`infrastructure/kraken/recognizer.py`);
* `CharNGram` (`infrastructure/lm/char_ngram.py`) is an order-6 character model
  with Stupid-Backoff smoothing. Every context is packed into one `int64` and
  the levels are stored as sorted arrays queried with `np.searchsorted`, so a
  27.6 M-character corpus costs 22 MB of RAM / 5.6 MB on disk instead of the
  gigabytes a Python dict needs;
* score = `log P_ctc + alpha * log P_lm + beta * words + word_bonus * known`
  (all log10; `beta` and `word_bonus` are deliberately small and tunable);
* the codec emits **NFD**, so the LM and the lexicon are built in NFD too —
  otherwise `й` (which arrives as `и` + U+0306) would be an unknown character.

### What the language model is built from

`scripts/build_htr_lm.py` mixes three sources and records their sizes in the
artifact, because not all of them teach the same thing:

| source | what it teaches |
| --- | --- |
| the author's confirmed pages, `--author-weight 30` | vocabulary, names, dialect — *and* what follows a space |
| running Russian prose (`--extra-text FILE`, or files in `htr_storage/lm/texts/`) | cross-word statistics: only real text teaches word boundaries |
| the word-form lists already used by the dictionary (`*russian*.txt`, ~2.4 M forms) | within-word letter sequences, surnames, toponyms |

Two measured traps, both of which turned a large win into a loss:

* **bare word forms have no spaces.** Feeding them as word-list lines without
  inventing sentences was the only variant that helped. Adding synthetic
  10-word "sentences" created 28 M characters of fake cross-word statistics
  that drowned the 0.3 M characters of the author's real text: WER 0.37 against
  greedy's 0.27. Spacing must come from real text only — hence the separate
  `running_text_chars` counter in the artifact.
* **the evaluated page must not be in the model.** An artifact built from all
  pages (page included, ×30) made page 20 look almost perfect — CER 5.64 % →
  1.32 %, WER 25.5 % → 5.7 % — while the leakage-free numbers below are the
  real ones. `scripts/measure_htr_decoders.py --chronological` rebuilds the
  model per page from the pages confirmed *before* it and caches the folds
  (`--leave-one-page-out`, which hides only the page itself, is the weaker
  variant). The chronological order is the honest one: when a page is
  recognized, the later pages have not been written down yet, so their words
  cannot be in the model. That is why the first page has no usable model at all
  and the win grows page by page.

`--per-author` writes one artifact per author (`author_<id>_char_lm.npz`), which
is what the recognizer picks up for that author; the global `ru_char_lm.npz`
stays the fallback for authors who have no confirmed pages yet (and, having no
running text of its own, it keeps them on greedy decoding).

**Production path check.** The shipped artifacts are exactly that pair:
`author_1_char_lm.npz` (9 258 characters of the author's own confirmed text)
plus the author's vocabulary as the word bonus. Decoding page 20 again through
the wired path (`KrakenRecognizer._build_decoder`, real codec, real Bloom
dictionary, real author words) gives CER 2.73 % / WER 11.46 % against greedy
5.64 % / 25.52 %. These numbers are deliberately **not** in the table below:
this is a re-recognition of an already confirmed page, so the language model
legitimately contains its text, and the author's own words are known. It only
proves that the wired path runs end to end and that the vocabulary reaches the
decoder — the honest estimate is the chronological table.

### How much the LM helps depends on the acoustic model (measured)

Everything above was measured with the **base** acoustic model, which has never
seen this handwriting. In production the author's own fine-tuned model runs, and
the language model then has far less to repair. The clean comparison is on two
pages the fine-tuned models have never seen (23 and 24, uploaded after v13 was
trained), same character-LM fold, same beam parameters, only the acoustic model
differs:

| acoustic model | CER vs stored text | WER vs stored text | **characters the LM changed** | lines changed |
| --- | --- | --- | --- | --- |
| base | 0.0516 | 0.1842 | **2.51 %** | 37 of 58 |
| v12 | 0.0290 | 0.0974 | **0.82 %** | 18 |
| **v13** | **0.0249** | **0.0763** | **0.73 %** | 14 |

The first two columns are *disagreement* with what the pipeline produced for
those pages (they have no user corrections yet, so there is no ground truth);
the third column is reference-free — it is the edit distance between the same
model's greedy and beam output. As the acoustics get sharper the LM's
intervention shrinks **3.4×**, which is exactly the expected effect: fewer
ambiguous characters left to fix. The word-level second pass is even smaller —
0.05 % of characters on v13 — and on those two pages it slightly *worsened* WER
(0.0763 → 0.0789), which is why it ships disabled.

### Re-tuning the weights on strict ground truth

Two methodological fixes were needed before the weights could be re-tuned:

* **honest acoustic folds** — every page is recognized by the model that existed
  *before that page was confirmed* (v2/v7/v8/v9/v10/v11/v12 for pages 3, 9, 14,
  16, 17, 18, 19, 20, 22). Using v12 on pages it was trained on would measure
  memorization, not recognition. `--model-map` takes the mapping and the matrix
  cache is keyed by model, so folds never mix;
* **strict ground truth** — `--only-corrected` counts only the lines the user
  actually corrected. Without it the reference of an untouched line is the
  previous model's own output, and the LM is punished for disagreeing with it.
  That is what had produced the old α = 0.1: on the strict reference 0.1 is
  consistently worse.

Grid on 142 corrected lines, honest folds (CER / WER):

| acoustic group | α = 0.0 | α = 0.05 | α = 0.1 (old) | α = 0.2 | α = 0.4 |
| --- | --- | --- | --- | --- | --- |
| weak (v2: pages 3, 9, 14) | .1047/.3342 | .0964/.3045 | .0887/.2722 | **.0807/.2488** | .0799/.2550 |
| medium (v7–v9) | .0549/.2167 | .0517/.2031 | .0493/.1920 | **.0460/.1760** | .0460/.1754 |
| strong (v10–v12) | .0324/.1484 | .0312/.1415 | .0300/.1342 | **.0240/.1053** | .0240/.0987 |

α = 0.2 wins in **every** group — the weight does not have to shrink on sharper
acoustics; if anything the opposite. Re-tuning the rescoring weight on the same
strict reference (leave-one-page-out selection, two α values, guard on/off,
interpolation weights 1.0/0.0 and 0.7/0.3) gives the whole pipeline:

| configuration | CER | WER |
| --- | --- | --- |
| α = 0.1, no second pass (what production ran) | 0.0653 | 0.2212 |
| **α = 0.2**, no second pass | 0.0593 | 0.1998 |
| **α = 0.2 + second pass** (w = 1.0, guard off, general-only word model) | **0.0577** | **0.1957** |

So the re-tuning is worth **−11.5 % relative WER**, of which the second pass
contributes only the last 2 % — and it selects `guard=False` and `W_author=0` on
almost every page, i.e. the guard and the author half both stay out of the way.

```
.venv/bin/python scripts/measure_htr_decoders.py --pages 3 9 14 16 17 18 19 20 22 \
    --model-map htr_storage/lm/model_folds.json --chronological --fold-tag chrono_leipzig \
    --reuse-cache --only-corrected --alpha 0.2 --json-out htr_storage/lm/alpha_grid_a0.2.json
```

### A real corpus is the biggest single lever (measured)

The three sources above are not equal. Word-form lists carry the letter
patterns; **only running text teaches what follows a space**, and the author's
own 9 k characters are far too little. Feeding the model a real corpus is a
one-flag change (`--extra-text FILE`, or drop files into
`htr_storage/lm/texts/`) and it was measured with exactly the same protocol,
matrices and parameters as the table below — only the corpus differs:

| | beam CER | beam WER | vs greedy WER |
| --- | --- | --- | --- |
| word forms + author text (9 k chars of running text) | 0.0632 | 0.2297 | −15 % |
| **+ 40 M characters of Leipzig news text** | **0.0582** | **0.2101** | **−22 %** |

Per page the gain is not uniform, and the pattern is the interesting part:

| page | running text before | beam WER before | beam WER after |
| --- | --- | --- | --- |
| 3 | 0 | 0.3015 | **0.2764** |
| 9 | 1 157 | 0.2388 | 0.2488 |
| 14 | 2 356 | 0.1700 | **0.1300** |
| 16 | 3 512 | 0.2539 | **0.2332** |
| 17 | 4 648 | 0.2461 | **0.2356** |
| 18 | 5 772 | 0.2701 | **0.2322** |
| 19 | 6 975 | 0.1869 | **0.1566** |
| 20 | 8 094 | 0.1667 | 0.1667 |

The first page — the one that previously had *no* usable model and fell back to
greedy — improves the most, and the two regressions are small. The corpus used
here is the Leipzig Corpora news collection (`rus_news_2020_1M`, 1 M sentences,
102 M characters); 40 M characters is the builder's default `--char-budget`, and
raising it is the cheapest remaining experiment.

### Measured result (chronological folds, default model, α = 0.1)

Reused CTC matrices, beam width 32, top-k 8, greedy against beam on the *same*
matrices, reference = the user's own transcription:

| page | earlier pages in LM | greedy CER | beam CER | greedy WER | beam WER |
| --- | --- | --- | --- | --- | --- |
| 3 | 0 | 0.0949 | 0.0949 (greedy fallback) | 0.3015 | 0.3015 |
| 9 | 1 157 | 0.0654 | **0.0628** | 0.2438 | **0.2388** |
| 14 | 2 356 | 0.0612 | **0.0532** | 0.1900 | **0.1700** |
| 16 | 3 512 | 0.0884 | **0.0758** | 0.2850 | **0.2539** |
| 17 | 4 648 | 0.0896 | **0.0686** | 0.3298 | **0.2461** |
| 18 | 5 772 | 0.0776 | **0.0639** | 0.3033 | **0.2701** |
| 19 | 6 975 | 0.0697 | **0.0514** | 0.2576 | **0.1869** |
| 20 | 8 094 | 0.0564 | **0.0353** | 0.2552 | **0.1667** |
| **всего (239 строк)** | | 0.0753 | **0.0632** | 0.2707 | **0.2297** |

Beam wins on every page that has enough running text behind it, and the gain
grows with the corpus: −2 % relative WER on page 9, −35 % on page 20, and
**−15 % relative WER overall** (0.2707 → 0.2297) because the early pages barely
have a model. With 0 characters of running text (the very first page an author
confirms) the model knows letters but not spacing and decoded slightly *worse*
than greedy, so `htr_lm_min_text_chars` (default 1000) forces greedy below that
threshold — the first page is unaffected and the decoder gets better from
there. The check runs on `running_text_chars`, not on the
total corpus size: a model made only of word forms has millions of characters
and still cannot judge a space.

**Both fold protocols agree.** The same run with the objective's literal
protocol — `--leave-one-page-out`, which hides only the evaluated page — gives
the same overall result on the same matrices: greedy CER 0.0753 / WER 0.2707 →
beam CER 0.0622 / WER 0.2290 (+0.0131 CER, +0.0417 WER), against +0.0121 /
+0.0410 for the chronological folds. They differ exactly where they should: with
leave-one-page-out the model of the *first* page still contains the later pages
(8 100 characters), so it is usable there and that page improves too (CER
0.0949 → 0.0851). Page 9 is the only regression of either run (WER 0.2438 →
0.2587). The chronological numbers are quoted above because they describe what
the system can actually have at the moment it recognizes a page.

| setting | default | meaning |
| --- | --- | --- |
| `htr_decoder` | `beam` | `greedy` (fast, baseline) or `beam` |
| `htr_lm_path` | `htr_storage/lm/ru_char_lm.npz` | global fallback LM; per-author `author_<id>_char_lm.npz` in the same directory is preferred |
| `htr_beam_width` | `32` | prefixes kept per step (decode ≈ 1.1 s/line at 32, ≈ 2.7 s at 60) |
| `htr_beam_top_k` | `8` | characters expanded per prefix |
| `htr_beam_alpha` | `0.2` | LM weight; re-tuned on strict ground truth (see the acoustic-strength section) |
| `htr_beam_beta` | `0.0` | per-word length bonus |
| `htr_beam_word_bonus` | `0.8` | bonus per word known to the lexicon |
| `htr_lm_min_text_chars` | `1000` | below this much running text the decoder stays greedy |
| `htr_word_lm_path` | *(empty)* | word-level KenLM for the second pass; empty or a missing `kenlm` disables it (single-pass decoding). Measured to be noise-level, so it ships off — see the second-pass section |
| `htr_rescore_weight` | `0.5` | weight of the word model: `final = beam_score + w · word_score` |
| `htr_rescore_n` | `10` | hypotheses the beam hands to the second pass |
| `htr_rescore_guard` | `false` | allow a re-ranking only when the top-1 has an out-of-dictionary word or is unsure (measured a wash; kept for a sharper future model) |
| `htr_rescore_min_mean_acoustic` | `-0.30` | mean acoustic log10 per character below which the line counts as unsure |
| `htr_word_alternatives` | `5` | how many alternative readings to keep per word for the editor (`0` = off) |

The fallback is silent by design: a missing artifact, no space in its alphabet,
or too little running text logs one warning and decodes greedily, so a page
always comes back. The decoder is chosen per recognition call, and the beam
result is stored as `predicted_text` exactly like a greedy one — the language
model never writes into `corrected_text`, and every accepted or rejected
proposal is logged in `HTRSuggestionEvent` (see point 17 of the overview).

## Word-level KenLM model (two-pass decoding)

The character n-gram is deliberately cheap: it is asked ~52 000 times per line
and ~1.5 M times per page, so it cannot afford word order. The second pass can:
the beam hands over its N best hypotheses (`PrefixBeamSearch.decode_nbest`),
and a **word**-level KenLM model re-ranks them, where word choice and word
order are visible (`infrastructure/lm/word_rescorer.py`).

### Building the model (KenLM's own tools, no hand-rolled weighting)

`scripts/build_htr_word_lm.py` runs the whole pipeline: two `lmplz` estimations
in intermediate format (general corpus + the author's own pages),
`interpolate --just_tune` for the weights, `interpolate` for the final model and
`build_binary` for the mmap-able artifact. KenLM has to be built first — it is
not on PyPI as a wheel, and the sdist ships C++ generated by an old Cython that
does not compile on Python 3.13:

```bash
# Boost is required by lmplz/interpolate; if the distro packages are missing,
# build the three needed libraries into a prefix you own (no root needed):
curl -sL -o boost.tar.gz https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz
tar xzf boost.tar.gz && cd boost_1_83_0
./bootstrap.sh --with-libraries=program_options,thread,system --prefix=/tmp/boost-install
./b2 -j4 install

# Eigen is needed by interpolate (header-only, but CMake needs its config):
cmake /tmp/eigen-3.4.0 -DCMAKE_INSTALL_PREFIX=/tmp/eigen-install && make install

# KenLM itself (the sdist's python/kenlm.cpp must be regenerated on 3.13)
cython --cplus -3 -o python/kenlm.cpp python/kenlm.pyx
cmake . -DBOOST_ROOT=/tmp/boost-install -DBoost_NO_SYSTEM_PATHS=ON \
        -DEigen3_DIR=/tmp/eigen-install/share/eigen3/cmake -DCMAKE_INSTALL_PREFIX=/tmp/kenlm-install
make -j4 && make install
```

Then, from `backend`:

```bash
.venv/bin/python scripts/build_htr_word_lm.py \
    --general-corpus /path/to/rus_news_2020_1M-sentences.txt --author-id 1 \
    --weights 0.5 0.5 --kenlm-bin /tmp/kenlm-install/bin
# one small author model per evaluated page, for an honest measurement
.venv/bin/python scripts/build_htr_word_lm.py --author-arpa-only --exclude-page 20 --output author_20.arpa
```

### Measured result: what the second pass actually buys (honest)

Same chronological folds, same matrices, same beam parameters; the word model is
the general Leipzig model (its author half is measured useless at this corpus
size, see below), weight 0.5, N = 10:

| | beam | + rescoring |
| --- | --- | --- |
| CER, weight chosen on the other pages (leave-one-page-out) | 0.0582 | **0.0579** |
| WER, same | 0.2101 | **0.2082** (+0.9 %) |
| CER / WER, best weight chosen on all 8 pages (optimistic) | 0.0582 / 0.2101 | **0.0571 / 0.2032** (+3.3 %) |

So the second pass is a **small** win on this corpus, not a large one — and the
first measurement said otherwise for an instructive reason.

> **The trap: an interpolated KenLM model cannot be un-mixed at query time.**
> The first tuning run scored candidates with the *already interpolated*
> (general 0.5 + author 0.5) binary as its "general" model and then added the
> author's ARPA on top of it. Because the author's own pages were baked into
> that binary, lines from those pages scored suspiciously well: WER appeared to
> drop from 0.2101 to **0.1464**, the optimum weight ran away to the edge of the
> grid, and the guard looked actively harmful. With two *pure* components
> (general as one model, a per-page author ARPA built without that page as the
> other) the same grid gives +1…3 % and the optimum sits at a weight of ~0.5.
> This is the same leakage that already invalidated one character-LM table
> earlier in this document — a page must never be scored by a model that has
> seen it.

The changes the second pass makes are the ones a character model cannot see
(word boundaries, endings, `годе → года`), but on 239 lines they move about one
word in five hundred. The infrastructure is in place and one setting turns it
off; the honest recommendation is to re-tune it when there is more author text,
or to replace the word model with a stronger one (a real corpus for the author
half, or a neural re-ranker).

### Two findings worth knowing

* **`lmplz` refuses to build the author half without `--discount_fallback`.**
  With 235 lines / 1 581 tokens / 991 types, modified Kneser-Ney estimates a
  negative discount and the tool aborts: *"2-gram discount out of range for
  adjusted count 3: -0.79 ... rerun with --discount_fallback"*. The fallback is
  KenLM's documented escape hatch for tiny corpora and the script always passes
  it for the author model (the general model does not need it).
* **`interpolate --just_tune` returns `nan` on the author's own text.** 235
  sentences are not enough to tune two weights by likelihood; the resulting ARPA
  contains `NaN` and `build_binary` refuses it. The script detects this and
  either fails loudly or uses explicit `--weights`. The honest way to pick the
  weights here is a grid on the *decoding* metric (leave-one-page-out), the same
  way α was tuned — the tuning data is far too small for a likelihood criterion.

### The guard: what it is for, and what it measured

A word-level model is trained on general news text, so it pulls towards frequent
words; applied without care it can trade «Судиславль» for a common word. The
guard in `WordRescorer` allows a re-ranking **only** when the first pass is
unsure — the top-1 contains a word the dictionary does not know, or its mean
acoustic confidence is below `min_mean_acoustic`. A confident, fully known top-1
is kept and the word model is not even queried.

Measured on the honest (pure-component) grid the guard is a wash: the best
guarded configuration gives WER 0.2044, the best unguarded 0.2032, and the
leave-one-page-out selection chooses unguarded on most pages. It is kept as an
option (`htr_rescore_guard`) rather than a default because it costs nothing to
have and protects a case the corpus is still too small to show — a rare name
inside an otherwise confident line.

For the measurement the two halves are kept apart (the general binary plus a
per-page author ARPA built without that page's text) and combined with the same
weights; production uses the single interpolated binary. **Both inputs must be
pure** — see the trap above.

## Provisioning (WSL2 / Linux)

`coremltools` (needed by kraken for its CoreML format) has Linux wheels, so a
full install works here — unlike the earlier Windows deployment:

```bash
cd backend
uv pip install --python .venv/bin/python kraken==7.1.1
# kraken pins safetensors~=0.7.0 while transformers (imported by torchmetrics)
# requires safetensors>=0.8.0. kraken 7.1.1 works with 0.8.0, so force it back:
uv pip install --python .venv/bin/python safetensors==0.8.0
```

If `safetensors` stays at 0.7.x, `import transformers` fails and with it
`lightning`/`torchmetrics`, which breaks the training backend.

CUDA: training and (optionally) recognition use the torch build in the venv.
A CUDA-enabled torch must be installed, e.g. `torch==2.14.0+cu130` for an RTX
50-series card. Verify with:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Model

```bash
cd backend
.venv/bin/python scripts/download_htr_model.py            # medium, ~61 MiB
```

Downloads the multilingual PP-OCRv6 line recognizer (kraken author, Apache-2.0)
into `htr_storage/models/default/ppocrv6_medium.safetensors`, which is what
`htr_default_model_path` points at, and verifies the published checksum.
`htr_storage/` is git-ignored: models are provisioned, never committed.

Individual authors are switched to their own fine-tuned model through the model
lifecycle (`ACTIVE` version of the author wins over the default).

## Training

`HandwritingTrainingService` orchestrates `KrakenTrainer`, which targets the
kraken 7 training API (`kraken.train`, config objects, safetensors output) and
mirrors `ketos train`:

```text
training data: all CONFIRMED pages of the author -> line crops + transcripts
      |
      v
PPOCRv6RecognitionModel.load_from_weights(default model)   # fine-tune --load
      |
      v
KrakenTrainer.fit()  (checkpoint with the best val_metric)
      |
      v
convert_models(best checkpoint -> <author>/vN/model.safetensors)
      |
      v
trainer.test() on the held-out validation split -> CER / WER
```

Business rules enforced by the application layer (see
`application/training_service.py` and its tests):

* the base model is **always** the default model, never a previous author model;
* every fine-tune is built from **all** confirmed pages of the author;
* each successful run creates a new `ModelVersion` (`v1`, `v2`, …); older
  versions are kept and only one version per author is `ACTIVE`;
* a version is activated only after training *and* metrics extraction succeed;
  on failure the previous `ACTIVE` model stays in place;
* **a fine-tune is activated only if it beats the base model** on the same
  validation split (primary criterion: CER). Otherwise the run returns
  `NO_IMPROVEMENT`, the version is marked `FAILED` and the previous model stays
  active — see below;
* the user's `corrected_text` is ground truth and is never normalized or
  spell-fixed (`normalize_whitespace=False` in the kraken data config).

### Training crops follow the line, not its box

Every training sample is a dewarped strip cut along the line's baseline, using
kraken's own extractor (`kraken.lib.segmentation.extract_polygons`) — the same
function recognition uses. `infrastructure/kraken/lines.py` rebuilds a
baseline + bounding polygon from the stored outline (the segmenter's boundary)
and hands it to kraken; the plain box crop remains as the fallback for lines
without an outline and as a safety net.

The effect is measurable on the very same 8 validation lines, with the very same
base model — only the crop geometry differs:

| crops | base model CER | fine-tuned CER |
| --- | --- | --- |
| axis-aligned boxes | 0.2117 | 0.1227 |
| **baseline strips** | **0.0951** | **0.0613** |

The base column is the interesting one: nothing but the input representation
changed, and the recognizer's error dropped by more than half. It also fixes a
data bug — the box of a merged line can be degenerate (one page had a *6 px
wide* box around a full sentence), and such a sliver went into training as-is.

### Line height: 96 px is what the base model actually uses

PP-OCRv6 was pretrained at a line height of 128 px, and this project's
`htr_training_height` was 96, which looked like an accidental scale mismatch.
It is not one: the exported checkpoint declares its own input shape
(`input=(1, 3, 96, 0)`), and kraken overwrites `config.height` with the
checkpoint's value when the model is loaded (`train/ppocr.py`), so
`htr_training_height` never reached the network in the first place. Measured
with the same corpus, split and seed:

| run | requested | effective | val CER | val WER | base CER |
| --- | --- | --- | --- | --- | --- |
| A (ad-hoc) | checkpoint | 96 | 0.0303 | 0.1304 | — |
| A | 128 | 128 | 0.0325 | 0.1366 | — |
| B (`scripts/measure_htr_height.py`) | checkpoint | 96 | 0.0337 | **0.1366** | 0.0685 |
| B | 128 | 128 | 0.0337 | 0.1429 | 0.0685 |

Run-to-run spread from stochastic training (augmentation, CUDA reduction order)
is about 0.003 CER — the same order as the difference itself — so the honest
reading is: **128 is never better** (WER is worse in both runs, CER is equal in
one and worse in the other), and the checkpoint's own 96 is kept. Run B is the
reproducible one: the identical base CER of 0.0685 across both heights confirms
the split and corpus were identical.

Two mechanics matter here. First, the height only changes through the explicit
`htr_training_height_override` backend option; setting `htr_training_height`
alone does nothing, because kraken rewrites `config.height` from the checkpoint
after loading. Second, the trainer reports the height it really used in
`training_metrics.line_height` (plus `line_height_override_from` when it forced
one), and the training banner in the UI shows it, so a future run cannot
silently train at an unexpected scale. The default override is `0` (trust the
checkpoint), which is the configuration the table above validates.

`htr_training_linetype` (default `baselines`) declares the crops to kraken as
baseline strips; with `bbox` the old behaviour and the
"trained on baselines ... but training set is bbox" warning come back.

### What made fine-tuning useless (and what fixed it)

The first real fine-tunes on this corpus all came out **worse** than the base
model (CER 0.38, 0.21, and finally 0.2117 vs a 0.2117 baseline — the exported
model *was* the base model). An A/B on the same 87-line corpus and the same
8-line holdout found two causes, both in kraken's ppocrv6 recipe rather than in
this project's wiring:

| setting | val CER | vs base 0.2117 |
| --- | --- | --- |
| kraken recipe as-is (aux NRTR head, BatchNorm in train mode, lr 1e-5) | 0.2117 | no change |
| + auxiliary NRTR head disabled, BatchNorm statistics frozen | 0.1595 | −25 % |
| + learning rate 1e-4 | **0.1104** | **−48 %** |

**1. The auxiliary NRTR head.** kraken's ppocrv6 training minimizes
`ctc_loss + nrtr_loss`, where the NRTR head is a 4-layer transformer decoder
(18.9 M parameters — more than the recognizer's own 15.9 M) used **only during
training**: recognition decodes the CTC output. An exported inference checkpoint
(`ppocrv6_medium.safetensors`) contains no such head, so kraken builds it from
random weights and gives it loss weight 1. The log shows the consequence
precisely — the model is fine while the backbone is frozen (val accuracy 0.79)
and collapses to 0.32 in the first epoch after unfreezing, because the features
are now serving a random objective. `htr_training_aux_nrtr = false` (default)
swaps in `infrastructure/kraken/modules.py::CtcFineTunePPOCRv6`, which freezes
the head before kraken's parameter grouping runs (so it is not even in the
optimizer) and returns a zero auxiliary loss.

**2. BatchNorm statistics.** The backbone has 61 `BatchNorm2d` layers and an
epoch has 20 steps of 4 line crops. Re-estimating those statistics from such
batches destroys a model that was previously fine.
`htr_training_freeze_bn = true` (default) keeps every BatchNorm in eval mode for
the whole run while its weights still learn.

The learning rate was a red herring: 1e-5 was introduced earlier to stop the
collapse, but the collapse came from the two causes above, and at 1e-5 the model
barely moves in 200 steps. It is now 1e-4.

### Freezing the backbone

`htr_training_freeze_backbone` keeps everything but the codec projection frozen
for the first steps of a fine-tune and only then lets the pretrained features
move. On a small corpus the freshly resized output layer (a new codec means
randomly initialized weights) otherwise pushes large gradients into the
backbone and the model forgets what it knew — that is what made this project's
first fine-tune worse than the base model.

| value | meaning |
| --- | --- |
| `-1` (default) | auto: freeze for the first epoch (`ceil(train_samples / batch_size)` optimizer steps) |
| `0` | off — the whole network learns from step 0 |
| `N > 0` | freeze for exactly N optimizer steps |

**The value is in optimizer steps, not samples.** kraken's CLI help says
*samples*, but its callback compares against `trainer.global_step`, so samples
would silently freeze the backbone for the whole run of a small corpus. The
number actually used is recorded in the model version's metrics as
`freeze_backbone_steps` (with `steps_per_epoch` next to it), and a run that
would never unfreeze is logged as a warning.

**Why this is not kraken's own `--freeze-backbone`**: `KrakenFreezeBackbone`
freezes `pl_module.net[:-1]`, which only works for the VGSL family (an
`nn.Sequential`). This project fine-tunes PP-OCRv6, whose net is a plain
`nn.Module`:

```console
$ python -c "from kraken.train.ppocr import PPOCRv6Model as M; M(variant='tiny')[:-1]"
TypeError: 'PPOCRv6Model' object is not subscriptable
```

So `infrastructure/kraken/callbacks.py` reimplements it against kraken's
`_output_proj` property (the codec-sized CTC projection), with two further
differences: unfreezing triggers on `global_step >= N` instead of `== N` (a
skipped step must not leave the backbone frozen for the whole run), and the
parameters stay in the optimizer's parameter group, so after unfreezing they
continue with the configured learning rate and schedule rather than a new group
at `lr / 10`.

### Quality gate (why `NO_IMPROVEMENT` exists)

Fine-tuning a 15.9 M-parameter model on ~60 lines can easily make it *worse*.
The first real run on this project (10 epochs, LR 1e-4) did exactly that:
`best_val_accuracy` was already the epoch-0 value, and on the shared validation
split the fine-tune scored CER 21.6 % / WER 46.7 % against the default model's
CER 14.6 % / WER 20.0 %. Publishing it would have degraded recognition.

`KrakenTrainer` therefore scores **both** the fine-tuned model and the untouched
base model on the identical held-out split and returns `baseline_metrics`
alongside `validation_metrics`; `HandwritingTrainingService` compares them and
refuses to activate a regression. `scripts/compare_htr_models.py` runs the same
comparison on demand:

```bash
.venv/bin/python scripts/compare_htr_models.py --author-id 1
```

Small corpora also use a gentler default (`htr_learning_rate=1e-5`,
`htr_training_warmup=10`, `htr_batch_size=4`); as more pages are confirmed the
fine-tune starts to win and the gate lets it through.

### Text representation: NFC on storage, NFD for training

The recognizer normalizes predictions to **NFC** (`KrakenRecognizer`), because
the model decodes Cyrillic as decomposed sequences (`и` + U+0306) and storing
that raw would make corrections and CER/WER count combining marks separately.

The default model's **codec**, however, contains the decomposed form and has no
`й`/`ӗ` code points. Feeding NFC text back as training ground truth therefore
looks like brand-new symbols: kraken logged *"Resizing codec to include 2 new
code points"* and grew the output layer 1623 → 1625 with **randomly initialized
weights**, which perturbs the CTC head. `htr_training_normalization=NFD`
(kraken's own recommended default) decomposes the training text again, so the
alphabet matches the codec and no output layer is rebuilt. This is canonical
equivalence, not a spelling/whitespace fix: the user's text is still used
verbatim in every other respect.

### Rolling back a model

Every version is kept, so a bad activation is reversible:

```bash
.venv/bin/python scripts/set_active_htr_model.py --author-id 1 --list
.venv/bin/python scripts/set_active_htr_model.py --author-id 1 --default    # use the default model
.venv/bin/python scripts/set_active_htr_model.py --author-id 1 --version 3  # re-activate a version
```

### When does training start?

Training is a **manual, explicit** step: the user presses «Обучить модель» in the
«Рукописи» tab, which calls `POST /htr/authors/{id}/train`. Confirming a page
only turns it into ground truth — otherwise confirming several pages in a row
would pay for a full fine-tune (minutes) after each one.

A fine-tune is still skipped while the author's confirmed corpus is too small to
be worth it. A single page (~30 lines) is not, so the threshold is configurable:

| setting | default | meaning |
| --- | --- | --- |
| `htr_min_training_lines` | `50` | confirmed lines required before fine-tuning |
| `htr_min_training_words` | `0` | optional second threshold; `0` disables it |

Below the threshold the training call returns `outcome = "INSUFFICIENT_DATA"`
with `lines_collected`/`lines_required` (and `words_*`), so the UI can explain
what is still missing. No model version is created for a skipped run.

A second training request arriving while the same author is already training gets
`outcome = "BUSY"` immediately (a process-local per-author lock; a worker queue
replaces it later).

Because every run rebuilds the corpus from the confirmed pages that exist at
that moment, a page confirmed after the last run is simply not in that model —
press «Обучить модель» again to include it.

### Keeping bad pages out of training

`DELETE /htr/pages/{id}` removes the page row (lines and words cascade) plus its
image and line crops. Every fine-tune rebuilds the corpus from the confirmed
pages that exist *at that moment*, so a deleted page never reaches a later
training. Model versions trained earlier keep the `dataset_hash` they were
built from, so the removal stays auditable and already trained models are not
touched. The UI offers the same action per page (trash icon) with a warning
that a confirmed page leaves the training set.

### Metrics and holdout

Each run stores training and validation metrics in the model metadata. The
validation split is deterministic (`htr_random_seed`) and is **taken from the
author's own confirmed pages** — it is not an independent holdout, which the
metadata states explicitly (`holdout_used: false` plus a `note`). CER/WER are
computed by kraken's own test pass over that split; kraken reports character
and word *accuracy*, which the adapter converts to error rates.

### Real-backend smoke test

```bash
HTR_RUN_REAL_TRAINING=1 HTR_TEST_DEVICE=cuda:0 \
    .venv/bin/python -m pytest tests/test_htr_kraken_training_smoke.py -q -s
```

It renders a few line images, fine-tunes the real default model for one epoch,
exports safetensors and checks CER/WER (≈30 s on an RTX 5060 Ti with
`TORCHDYNAMO_DISABLE=1`). It is skipped unless `HTR_RUN_REAL_TRAINING=1`.

## Settings (`app/config.py`)

### Recognition

| setting | meaning |
| --- | --- |
| `htr_default_model_path` | default recognition model / fine-tuning base |
| `htr_recognition_device` | `cpu`, `cuda:0`, `auto`; if the configured GPU has no usable CUDA backend in this process, recognition logs a warning and runs on CPU instead of failing |
| `htr_recognition_batch_size` | lines per forward pass |
| `htr_recognition_padding` | blank padding left/right of each line crop |
| `htr_recognition_text_direction` | `horizontal-lr` for Latin/Cyrillic pages |
| `htr_recognition_num_line_workers` | `0` = in-process line extraction |
| `htr_segmentation_engine` | `neural` (bLLA baselines) or `classical` (straight boxes) |
| `htr_segmentation_merge_lines` | `true`: glue a detached first word back onto its line before recognition |
| `htr_segmentation_maxcolseps` / `htr_segmentation_no_hlines` | classical segmenter knobs (ignored by the neural one) |

Recognition is CPU-bound by default: the diary page used during development
(3472×4640) takes ≈2 minutes on CPU with the `medium` model, ≈0.9 s per line.
On a working CUDA setup `htr_recognition_device=cuda:0` reduces that to
seconds (the whole page — neural segmentation *and* recognition — took ≈50 s on
an RTX 5060 Ti), but it then shares the GPU with training. With the neural
segmenter the CPU path is noticeably slower, so set `cuda:0` when a GPU exists.

### Fine-tuning

| setting | default | meaning |
| --- | --- | --- |
| `htr_device` | `cuda:0` | training device (`cpu`, `cuda:0`, `auto`) |
| `htr_epochs` | `10` | epochs per fine-tune (`kosinus` schedule from the config) |
| `htr_min_epochs` | `0` | Lightning `min_epochs` |
| `htr_batch_size` | `4` | lines per training batch (raise only if VRAM allows) |
| `htr_learning_rate` | `0.00001` | initial LR; 1e-4 is what the measurements above use |
| `htr_validation_split` | `0.1` | share of confirmed lines held out for metrics |
| `htr_random_seed` | `42` | seeds the run and the deterministic split |
| `htr_min_training_lines` / `htr_min_training_words` | `50` / `0` | fine-tuning thresholds |
| `htr_training_resize` | `union` | codec handling when the data has new characters |
| `htr_training_normalization` | `NFD` | Unicode normalization of training text (must match the codec) |
| `htr_training_height` / `htr_training_max_width` | `96` / `2560` | line geometry (informational: the loaded checkpoint overrides the height — it declares 96, see the height section) |
| `htr_training_height_override` | `0` | force a different line height after loading (`0` = trust the checkpoint); A/B with `scripts/measure_htr_height.py` |
| `htr_training_variant` | `medium` | PP-OCRv6 size if training from scratch |
| `htr_training_precision` | `32-true` | Lightning precision |
| `htr_training_num_workers` | `0` | dataloader workers (`0` = in-process) |
| `htr_training_augment` | `true` | image augmentation (useful for small corpora) |
| `htr_training_schedule` / `htr_training_warmup` / `htr_training_weight_decay` | `cosine` / `10` / `0.01` | optimizer schedule |
| `htr_training_freeze_backbone` | `-1` | freeze the pretrained backbone for the first steps (`-1` = first epoch, `0` = off) |
| `htr_training_freeze_bn` | `true` | keep BatchNorm statistics frozen (batches are 4 line crops) |
| `htr_training_linetype` | `baselines` | how the crops are declared to kraken (`baselines` = dewarped strips, `bbox` = old boxes) |
| `htr_training_aux_nrtr` | `false` | use kraken's auxiliary NRTR decoder loss (off: the base checkpoint has no such head) |
| `htr_training_compile` | `false` | `torch.compile` (inductor) for the training step |
| `htr_training_matmul_precision` | `high` | TF32 matmuls on Tensor Core GPUs |

All values are plain settings, never hardcoded in the business logic.

### `torch.compile` is off by default

kraken wraps its training forward pass in `torch.compile` unconditionally.
On a small fine-tuning corpus that codegen pass costs more than the training
itself: the first run on this project's machine spent **>10 minutes of 100% CPU
with an idle GPU** while Inductor compiled the static-shape graph (≈550 MiB of
kernel cache), and the run looked hung. `htr_training_compile=false` sets
`TORCHDYNAMO_DISABLE=1` before kraken calls `torch.compile`, so training starts
immediately in eager mode. Enable it only for large corpora, where the kernel
speedup pays the compile cost back.

If a run is interrupted (backend restart, OOM, cancel), its `ModelVersion`
stays `TRAINING`; the next training for that author marks such versions `FAILED`
before creating a new one, so the version list never keeps a zombie.

## Architecture

```text
routes/htr.py                      HTTP only
  └── application/
        page_service.py            upload → recognize → edit → confirm
        lexicon.py                 dictionary annotation of the read models
        dataset_builder.py         all CONFIRMED pages → TrainingDataset (+hash)
        readiness.py               lines/words threshold → ready / INSUFFICIENT_DATA
        training_service.py        orchestration, versioning, activation rules
        metrics.py / confidence.py CER/WER, confidence levels
  └── domain/                      entities + interfaces (no framework imports)
  └── infrastructure/
        page_repository.py, model_repository.py, storage.py
        lexicon/                   Bloom dictionary, author corpus, layered check
        kraken/recognizer.py       HTRRecognizer (kraken)
        kraken/trainer.py          HTRTrainer (kraken 7)
```

The application layer never imports kraken; a second backend only needs a new
`HTRTrainer` (and eventually `HTRRecognizer`) implementation.

## API

| endpoint | purpose |
| --- | --- |
| `POST /htr/authors/{id}/pages` | upload a page image (EXIF orientation is applied). An optional `source_path` form field keeps the client-side path of a folder upload; the file name becomes the page label |
| `GET /htr/authors/{id}/pages` | page list (id, file name, stored order, status, line count, dates, prediction CER, dictionary misses) for the sidebar |
| `PUT /htr/authors/{id}/pages/order` | store the order the user dragged the pages into (`page_ids`, ids the client did not send keep their relative order at the end) |
| `PATCH /htr/pages/{id}/name` | rename the displayed file name of a page (drops the client-side `source_path`) |
| `POST /htr/pages/{id}/recognize` | **run recognition**, store lines/words |
| `POST /htr/pages/{id}/recognition-result` | import a result produced by an external tool (geometry is validated; it does not recognize anything) |
| `GET /htr/pages/{id}` | read page + lines + words + confidence levels + `in_lexicon` / `oov_count` / `lexicon_available` |
| `DELETE /htr/pages/{id}` | delete a page (with its image and crops); it leaves every future training corpus |
| `GET /htr/pages/{id}/image` | the raw page image (the frontend fetches it with the bearer token and renders it as a blob) |
| `PATCH …/lines/{line}/words/{word}`, `PUT …/lines/{line}` | corrections (`corrected_text` is the single source of truth) |
| `POST /htr/pages/{id}/suggestions` | ask the LLM for fixes; they are stored as proposals, the transcription is untouched |
| `PUT …/lines/{line}/suggestion`, `DELETE …/lines/{line}/suggestion` | accept / dismiss one whole proposal |
| `PUT …/lines/{line}/suggestion/changes/{index}` | accept a single proposed word, the rest of the proposal stays |
| `POST /htr/pages/{id}/suggestions/accept` | accept every proposal whose changes the dictionary confirms |
| `POST /htr/pages/{id}/confirm` | confirm ground truth and measure prediction CER/WER (never starts training) |
| `POST /htr/authors/{id}/train` | fine-tune the author's model on all confirmed pages |
| `GET /htr/authors/{id}/models` | model versions of the author |

Recognition *and* training are synchronous for now (the training service is
HTTP-agnostic and can move to a Celery worker unchanged; training a real corpus
already takes minutes, so a worker is the next step).

## Word geometry when the beam and greedy texts disagree

The beam decoder and kraken's own greedy decoder do not always tokenize a line
the same way: a diary line with ``где -то`` came back from the beam as
``где-то``. The per-character cuts describe the *greedy* text, so the line's
words used to be dropped wholesale in that case — a correct transcription with
no word boxes at all, which reads as an interface bug. Measured on page 26, two
of 29 lines were in exactly that state.

Now the display text and the greedy text are aligned **character by character**
(`domain/text.py::map_spans_to_reference`), and every word takes the reference
range behind it: a merge covers both greedy words, a split gives each half its
own slice, a letter change maps onto the changed character, and a word the beam
inserted has no geometry *of its own* while the rest of the line keeps theirs.
If not a single word can be built, the line says so in the log instead of
silently returning nothing.

Pages recognized before the fix keep their missing boxes (the geometry comes
from the CTC matrix, which is not stored). ``scripts/repair_htr_word_geometry.py
--page-id N`` re-runs recognition on the page image and inserts the missing
``htr_words`` rows, mapping the geometry onto the **stored** text so nothing the
user corrected is touched:

```bash
.venv/bin/python scripts/repair_htr_word_geometry.py --page-id 26 --dry-run
.venv/bin/python scripts/repair_htr_word_geometry.py --page-id 26
```

Records are matched to lines by geometry first and, when the page was segmented
differently, by text similarity (a record carries no ``bbox`` on this path, so
the text fallback is what does the work in practice). The repair was applied to
page 22 (line 23) and page 26 (lines 23 and 26) — after it, no line in the
corpus has a transcription without word boxes. Repaired lines have geometry but
**no** alternative readings: those come from the CTC matrix of the *stored*
recognition, which is gone.

## Alternative readings per word

Clicking a word on the page selects it and, in the line's card, shows the other
readings the recognizer considered for **that** word:

```
'вечерам' -> 'Вечерам', 'вегерам', 'веерам', 'вечером'
'девчонки' -> 'девгонки', 'девонки'
'а' -> 'А', 'я', 'и', 'о', 'Я'
'завидовали.' -> 'завидовали,', 'завидовали:', 'заведовали.', 'завидовал.'
```

They come for free from the N-best list the beam search already produces
(`PrefixBeamSearch.decode_nbest`): the hypotheses that differ in one word
contribute their variants, aligned with `difflib.SequenceMatcher` so a merge
(`на зывалось` → `называлось`) or a split is offered as one alternative
(`domain/text.py::align_word_alternatives`). The list is written once, at
recognition time (`htr_words.alternatives`), because it derives from the CTC
matrix, which is not stored; clicking a variant inserts it through the same
save path a manual edit takes.

**The limitation is the point.** Only what the beam actually considered can
appear here — a word the acoustic model never produced (a rare surname, a
dialect spelling) cannot be in the list, however wrong the reading is. This is
not a replacement for the dictionary check (violet words) or for the LLM
proposals; it is the cheap "the model itself hesitated between these" layer.
More caveats, all measured:

* pages recognized **before** this feature have no alternatives (the list is not
  recomputed, because it comes from the CTC matrix, which is not stored);
* the list exists only when the beam is active (`htr_decoder=beam`) — greedy
  decoding has no hypotheses to compare;
* the **classical (bbox) segmenter path has no CTC matrix at all**. Measured on a
  real crop, `BBoxOCRRecord.logits` is a list of per-character tuples
  `(str, int, int, float)`, not a probability matrix, so that path decodes
  greedily and produces no alternatives. The neural (baseline) segmenter, which
  is the default, attaches the tensor and works; the recognizer now reports
  "no CTC matrix" and falls back cleanly instead of failing with a confusing
  "matrix must be 2-D" warning.

## Frontend: the «Рукописи» tab (`frontend/pages/manuscripts.vue`)

One screen for the whole page lifecycle: pick an author → upload → recognize →
correct → confirm. Layout is a three-column grid (page list, image with the word
overlay, line-by-line transcription).

* lines and words are drawn as an **SVG overlay of polygons** in page-pixel
  coordinates (falling back to the `bbox` rectangle for pages recognized before
  polygons existed); strokes are scaled by `1 / zoom` so they stay 1–2 px on
  screen at any zoom. The confidence colour comes from the API's
  `confidence_level` (`normal` / `warning` (< 0.90) / `critical` (< 0.70), both
  thresholds configurable) — grey outline, amber, red; the scan is light paper,
  so every polygon also carries a dark halo (drop-shadow);
* words the dictionary does not know are filled **violet** (a channel of their
  own, the confidence outline stays amber/red), the line gets a
  «не в словаре: N» badge and a chip per word — clicking a chip selects the word
  on the page. The sidebar shows each page's count and can sort by it
  («по словарю»); the selected word is outlined in white, so it is visible on
  top of every fill colour;
* clicking a word polygon scrolls to its line and selects the token in the
  textarea (only while the line's word alignment is still valid);
* transcriptions are per-line textareas; a line is saved on blur via
  `PUT /htr/pages/{id}/lines/{line_id}`. After a save the server marks
  `words_stale`, and the UI shows a «разметка устарела» badge and dims/dashes
  that line's polygons instead of pretending the old geometry still matches;
* «Предложить правки» shows the model's fixes *under* the line, as
  `было → стало` chips coloured by the dictionary verdict (green verified,
  violet unknown); each chip is clickable and applies only that word, while
  «Принять всё» / «Скрыть» work per line and «Принять проверенные (N)» works for
  the whole page — the textarea keeps showing the kraken output until something
  is accepted;
* «Распознать заново» also works for a page in `EDITING` (e.g. to pick up the
  new segmenter); it asks for confirmation because the corrections of that page
  are replaced (`?force=true`);
* the image viewer supports zoom (`−` / `+` / «по ширине» / 100 %, Ctrl+wheel)
  and drag panning; the page image is fetched with the bearer token and shown as
  an object URL, since `<img>` cannot send an `Authorization` header;
* «Обучить модель» (in the header, above the author selector) fine-tunes the
  author's model on all confirmed pages via `POST /htr/authors/{id}/train`; the
  outcome is shown inline (`SUCCESS` / `INSUFFICIENT_DATA` / `NO_IMPROVEMENT` /
  `BUSY` / `FAILED`) with CER/WER against the baseline model. Confirming a page
  never starts training by itself.
* the sidebar labels a page with the **name of the uploaded file** (not its id);
  «Загрузить папку» sends every image of a folder and keeps its relative path,
  so a page from `1975/` and one from `1976/` with the same file name are shown
  by their **full path** instead of the ambiguous bare name. Pages can be dragged
  into any order (the server stores `order_index`; sorting by «по словарю»
  disables dragging until it is switched off) and renamed with the pencil button
  (`PATCH /htr/pages/{id}/name`).

## Known remaining work

* The dictionary check highlights a whole word box on the page, not the offending
  characters inside the transcription textarea (that needs a contenteditable or
  a mirror layer behind the textarea). A word may also be a misspelling of a
  known word (`карова`), which no word list can flag.

* **The language model has too little text.** The beam decoder wins wherever it
  has running text, and the measurement shows the win growing with the corpus
  (−2 % WER on the first page with text, −35 % six pages later, −15 % over
  all eight pages) — so a real
  Russian corpus is the cheapest remaining gain. `scripts/build_htr_lm.py
  --extra-text FILE` (or dropping files into `htr_storage/lm/texts/`) is the
  only missing piece; the word-form lists alone cannot teach spacing.
* Beam was measured on the **base** acoustic model. Beam over the fine-tuned
  v12 model is untested (the cached matrices in `htr_storage/lm/matrices/` are
  from the base model; pass `--model` to re-recognize).
* The lexicon participates as a soft bonus, not as a hard constraint:
  lexicon-constrained decoding (a trie over the beam) is not implemented.
* `--per-author` artifacts are not built automatically. The recognizer prefers
  `htr_storage/lm/author_<id>_char_lm.npz` but nothing rebuilds it after a page
  is confirmed, so it currently falls back to the global model; rebuilding it
  could be hung off the training pipeline.
* Dataset preparation re-decodes the full page image for every line crop
  (`PilLineCropper.crop_line`), which costs ≈0.4 s per line — ≈30 s for a
  65-line corpus on the development page. Cropping all lines of a page from one
  decoded image would remove that.
* Nothing estimates how well a model does on a page it has never seen: every
  confirmed page goes into the corpus, so the gate compares a fine-tune against
  the base model on the author's *own* pages (`holdout_used` is always false).
  Reserving one confirmed page as a real holdout would make the gate objective.
* Training runs inside the HTTP request: move it to a Celery worker before
  corpora grow (the domain/application API does not change).

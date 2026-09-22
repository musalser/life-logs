# HTR module (handwriting recognition & per-author training)

Recognition pipeline: page image → binarization → line segmentation → CTC
recognition → editable text. Training fine-tunes a per-author model on the
confirmed lines of that author.

Dependencies point inward: `API → application → domain ← infrastructure`.
Kraken is imported **only** inside `infrastructure/kraken/` and only lazily, so
the application runs without the optional HTR backend (recognition then answers
`422` with an actionable message instead of crashing).

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

Fine-tuning runs after every confirmed page, but only once the author's
confirmed corpus is large enough for training to be worthwhile. A single page
(~30 lines) is not, so the threshold is configurable:

| setting | default | meaning |
| --- | --- | --- |
| `htr_min_training_lines` | `50` | confirmed lines required before fine-tuning |
| `htr_min_training_words` | `0` | optional second threshold; `0` disables it |

Below the threshold, confirmation still succeeds and the confirm response
returns `training.outcome = "INSUFFICIENT_DATA"` with
`lines_collected`/`lines_required` (and `words_*`), so the UI can explain when
the next fine-tune will happen. No model version is created for a skipped run.

A second confirmation arriving while the same author is already training gets
`outcome = "BUSY"` immediately (a process-local per-author lock; a worker queue
replaces it later).

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
| `htr_recognition_device` | `cpu`, `cuda:0`, `auto` (must exist in the torch build) |
| `htr_recognition_batch_size` | lines per forward pass |
| `htr_recognition_padding` | blank padding left/right of each line crop |
| `htr_recognition_text_direction` | `horizontal-lr` for Latin/Cyrillic pages |
| `htr_recognition_num_line_workers` | `0` = in-process line extraction |
| `htr_segmentation_maxcolseps` / `htr_segmentation_no_hlines` | classical segmenter knobs |

Recognition is CPU-bound by default: the diary page used during development
(3472×4640) takes ≈2 minutes on CPU with the `medium` model, ≈0.9 s per line.
On a working CUDA setup `htr_recognition_device=cuda:0` reduces that to
seconds, but it then shares the GPU with training.

### Fine-tuning

| setting | default | meaning |
| --- | --- | --- |
| `htr_device` | `cuda:0` | training device (`cpu`, `cuda:0`, `auto`) |
| `htr_epochs` | `10` | epochs per fine-tune (`kosinus` schedule from the config) |
| `htr_min_epochs` | `0` | Lightning `min_epochs` |
| `htr_batch_size` | `4` | lines per training batch (raise only if VRAM allows) |
| `htr_learning_rate` | `0.00001` | initial LR (gentle: 1e-4 forgot the pretrained model) |
| `htr_validation_split` | `0.1` | share of confirmed lines held out for metrics |
| `htr_random_seed` | `42` | seeds the run and the deterministic split |
| `htr_min_training_lines` / `htr_min_training_words` | `50` / `0` | fine-tuning thresholds |
| `htr_training_resize` | `union` | codec handling when the data has new characters |
| `htr_training_normalization` | `NFD` | Unicode normalization of training text (must match the codec) |
| `htr_training_height` / `htr_training_max_width` | `96` / `2560` | line geometry (variant overridden by the loaded model) |
| `htr_training_variant` | `medium` | PP-OCRv6 size if training from scratch |
| `htr_training_precision` | `32-true` | Lightning precision |
| `htr_training_num_workers` | `0` | dataloader workers (`0` = in-process) |
| `htr_training_augment` | `true` | image augmentation (useful for small corpora) |
| `htr_training_schedule` / `htr_training_warmup` / `htr_training_weight_decay` | `cosine` / `10` / `0.01` | optimizer schedule |
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
        dataset_builder.py         all CONFIRMED pages → TrainingDataset (+hash)
        readiness.py               lines/words threshold → ready / INSUFFICIENT_DATA
        training_service.py        orchestration, versioning, activation rules
        metrics.py / confidence.py CER/WER, confidence levels
  └── domain/                      entities + interfaces (no framework imports)
  └── infrastructure/
        page_repository.py, model_repository.py, storage.py
        kraken/recognizer.py       HTRRecognizer (kraken)
        kraken/trainer.py          HTRTrainer (kraken 7)
```

The application layer never imports kraken; a second backend only needs a new
`HTRTrainer` (and eventually `HTRRecognizer`) implementation.

## API

| endpoint | purpose |
| --- | --- |
| `POST /htr/authors/{id}/pages` | upload a page image (EXIF orientation is applied) |
| `POST /htr/pages/{id}/recognize` | **run recognition**, store lines/words |
| `POST /htr/pages/{id}/recognition-result` | import a result produced by an external tool (geometry is validated; it does not recognize anything) |
| `GET /htr/pages/{id}` | read page + lines + words + confidence levels |
| `PATCH …/lines/{line}/words/{word}`, `PUT …/lines/{line}` | corrections (`corrected_text` is the single source of truth) |
| `POST /htr/pages/{id}/confirm` | confirm ground truth, measure prediction CER/WER, then fine-tune (or report `INSUFFICIENT_DATA`) |
| `POST /htr/authors/{id}/train`, `GET /htr/authors/{id}/models` | manual training / model versions |

Recognition *and* confirmation/training are synchronous for now (training the
service is HTTP-agnostic and can move to a Celery worker unchanged; training a
real corpus already takes minutes, so a worker is the next step).

## Known remaining work

* Dataset preparation re-decodes the full page image for every line crop
  (`PilLineCropper.crop_line`), which costs ≈0.4 s per line — ≈30 s for a
  65-line corpus on the development page. Cropping all lines of a page from one
  decoded image would remove that.
* Line crops are axis-aligned boxes (`seg_type=bbox`) while the default model
  was trained on baselines; kraken warns about this. Inference uses the same
  bbox representation, so training and recognition stay consistent.
* Training runs inside the HTTP request: move it to a Celery worker before
  corpora grow (the domain/application API does not change).
* Page segmentation is the classical segmenter; multi-column or heavily skewed
  pages may need the neural bLLA model.

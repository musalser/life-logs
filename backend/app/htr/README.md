# HTR module (handwriting recognition & per-author training)

Recognition pipeline: page image → binarization → line segmentation → CTC
recognition → editable text. Training fine-tunes a per-author model on the
confirmed lines of that author.

Dependencies point inward: `API → application → domain ← infrastructure`.
Kraken is imported **only** inside `infrastructure/kraken/` and only lazily, so
the application runs without the optional HTR backend (recognition then answers
`422` with an actionable message instead of crashing).

## Provisioning on Windows (current deployment)

`coremltools`, which kraken needs for its CoreML model format, publishes **no
Windows wheels** — so `pip install kraken` cannot succeed on Windows. The
working setup is a kraken install without coremltools plus safetensors models:

```powershell
cd backend
.venv\Scripts\python.exe -m ensurepip --upgrade          # pip is not bundled
.venv\Scripts\python.exe -m pip install --no-deps kraken==7.1.1
.venv\Scripts\python.exe -m pip install `
    click "<8.3" jinja2 jsonschema lxml numpy packaging "pillow>=9.2.0" `
    platformdirs "protobuf>=3.0.0" pyarrow regex requests "rich<14.1.0" `
    "safetensors~=0.7.0" "scikit-image~=0.25.2" "scikit-learn~=1.7.2" `
    "scipy~=1.15.3" "shapely~=2.1.2" "threadpoolctl~=3.6.0" "torchmetrics>=1.1.0" `
    lightning==2.6.1 "htrmopo>=0.6,~=0.6" iso639-lang torchvision
```

Consequences of the missing coremltools, all handled in
`infrastructure/kraken/recognizer.py`:

* recognition models must be `safetensors` (`.mlmodel` files need the CoreML
  loader) — the adapter dispatches `kraken.models.load_safetensors` explicitly,
  because `kraken.models.load_models` aborts on the CoreML loader instead of
  skipping it;
* the neural bLLA segmenter (`kraken/blla.mlmodel`) is unusable, so the adapter
  uses the classical projection-profile segmenter `kraken.pageseg.segment`.

## Model

```powershell
cd backend
.venv\Scripts\python.exe scripts\download_htr_model.py            # medium, ~61 MiB
```

Downloads the multilingual PP-OCRv6 line recognizer (kraken author, Apache-2.0)
into `htr_storage/models/default/ppocrv6_medium.safetensors`, which is what
`htr_default_model_path` points at, and verifies the published checksum.
`htr_storage/` is git-ignored: models are provisioned, never committed.

Individual authors can be switched to their own fine-tuned model through the
model lifecycle (`ACTIVE` version of the author wins over the default).

## Settings (`app/config.py`)

| setting | meaning |
| --- | --- |
| `htr_default_model_path` | default recognition model / fine-tuning base |
| `htr_recognition_device` | `cpu`, `cuda:0`, `auto` (must exist in the torch build) |
| `htr_recognition_batch_size` | lines per forward pass |
| `htr_recognition_padding` | blank padding left/right of each line crop |
| `htr_recognition_text_direction` | `horizontal-lr` for Latin/Cyrillic pages |
| `htr_recognition_num_line_workers` | `0` = in-process line extraction (recommended on Windows) |
| `htr_segmentation_maxcolseps` / `htr_segmentation_no_hlines` | classical segmenter knobs |

Recognition is CPU-bound: the diary page used during development (3472×4640)
takes ≈2 minutes on CPU with the `medium` model, ≈0.9 s per line. Use
`--model small` for a faster, slightly less accurate variant, or
`htr_recognition_device=cuda:0` on a machine whose torch build has CUDA.

## API

| endpoint | purpose |
| --- | --- |
| `POST /htr/authors/{id}/pages` | upload a page image (EXIF orientation is applied) |
| `POST /htr/pages/{id}/recognize` | **run recognition**, store lines/words |
| `POST /htr/pages/{id}/recognition-result` | import a result produced by an external tool (geometry is validated; it does not recognize anything) |
| `GET /htr/pages/{id}` | read page + lines + words + confidence levels |
| `PATCH …/lines/{line}/words/{word}`, `PUT …/lines/{line}` | corrections (`corrected_text` is the single source of truth) |
| `POST /htr/pages/{id}/confirm` | confirm ground truth, measure prediction CER/WER, then train |
| `POST /htr/authors/{id}/train`, `GET /htr/authors/{id}/models` | manual training / model versions |

`POST /htr/pages/{id}/recognize` is synchronous for now (seconds per page, and
the service is HTTP-agnostic so it can move to a Celery worker unchanged).

## Known remaining work

* `KrakenTrainer` still targets the kraken 4-6 training API; kraken 7 uses
  `kraken.train.*` with config objects and safetensors output. Training
  therefore returns `FAILED` (previous active model is kept) until that adapter
  is ported. Fine-tuned author models must also be `safetensors` to be usable
  for recognition on Windows, so the trainer output path/naming
  (`model.mlmodel`) needs updating together with it.
* Page segmentation is the classical segmenter; multi-column or heavily skewed
  pages may need the neural bLLA model (Linux/macOS with coremltools, or
  converting it to a format kraken can load without coremltools).

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Live Journal AI"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/life_logs"
    ollama_url: str = "http://localhost:11434"
    llama_model_path: str = "ai_models/Qwen3.5-9B.Q4_K_M.gguf"
    llama_n_ctx: int = 4096
    llama_n_threads: int = 8
    port: int = 8000

    # NER
    ner_model: str = "surdan/LaBSE_ner_nerel"
    ner_min_score: float = 0.75

    # Auth
    jwt_secret_key: str = "change-me-in-env"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    refresh_cookie_name: str = "life_logs_refresh_token"
    refresh_cookie_secure: bool = False
    refresh_cookie_samesite: str = "lax"
    refresh_cookie_path: str = "/auth"
    refresh_cookie_domain: str | None = None
    auth_username: str = "admin"
    auth_password: str = "admin"

    # Celery / Redis
    # Use localhost defaults for local development; docker-compose overrides these to `redis` service host.
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # HTR (handwriting recognition & per-author model training)
    htr_storage_dir: str = "htr_storage"
    htr_default_model_id: str = "default"
    # Default recognition/multi-lingual handwriting model (kraken safetensors).
    # Fetch it with: python scripts/download_htr_model.py
    htr_default_model_path: str = "htr_storage/models/default/ppocrv6_medium.safetensors"
    htr_device: str = "cuda:0"

    # HTR decoding: how the CTC matrix becomes text.
    # 'greedy' = kraken's own decoder; 'beam' = prefix beam search with a
    # character n-gram language model (and a soft lexicon bonus), which fixes
    # letter-level confusions the acoustic model cannot resolve on its own.
    # Beam is the default because it was measured on leakage-free chronological
    # folds of the author's own pages: -15 % relative WER over eight pages
    # (0.2707 -> 0.2297), growing to -35 % on the page with the most text behind
    # it. Where that text is missing the recognizer falls back to greedy by
    # itself (htr_lm_min_text_chars), so the default is safe for a fresh author.
    # Reproduce with scripts/measure_htr_decoders.py --chronological --reuse-cache.
    htr_decoder: str = "beam"
    htr_lm_path: str = "htr_storage/lm/ru_char_lm.npz"
    htr_beam_width: int = 32
    htr_beam_top_k: int = 8
    # Weight of the language model, on a leakage-free model (built without the
    # page being decoded): at 0.1 the page with the most text behind it goes
    # from WER 25.5 % to 16.7 % and CER 5.6 % to 3.5 %; larger weights hurt
    # (0.45 -> 28.7 %). The weight is low because the model knows letters and
    # word boundaries well but has seen little running text.
    htr_beam_alpha: float = 0.1
    htr_beam_beta: float = 0.0
    htr_beam_word_bonus: float = 0.8
    # Beam decoding needs a language model with real running text behind it;
    # below this many characters of running text (author pages plus --extra-text,
    # *not* bare word forms) the model cannot judge word boundaries and greedy
    # decoding is used instead.
    htr_lm_min_text_chars: int = 1000

    # HTR recognition (inference)
    # Device string understood by kraken/lightning: 'cpu', 'cuda:0', 'auto'.
    htr_recognition_device: str = "cpu"
    htr_recognition_batch_size: int = 8
    htr_recognition_padding: int = 16
    htr_recognition_text_direction: str = "horizontal-lr"
    # 0 keeps line extraction inside the request process; >0 spawns workers,
    # which needs a working multiprocessing start method (unreliable on Windows).
    htr_recognition_num_line_workers: int = 0
    # Segmentation of the page into text lines.
    # 'neural' = bundled bLLA model: baseline polygons that follow curved lines
    # (loading its weights needs coremltools, available on Linux/WSL);
    # 'classical' = projection-profile segmenter producing straight boxes.
    htr_segmentation_engine: str = "neural"
    # The segmenter sometimes detaches the first word of a line into its own
    # record; glue such pieces back together before recognition.
    htr_segmentation_merge_lines: bool = True
    htr_segmentation_maxcolseps: int = 2
    htr_segmentation_no_hlines: bool = True

    # LLM proposals for the raw recognition (same Ollama instance as the chat).
    # A proposal never overwrites the transcription: it is stored separately and
    # the user accepts or dismisses it (POST /htr/pages/{id}/suggestions).
    # Best-effort: when Ollama is unavailable the line simply gets no proposal.
    htr_correction_enabled: bool = True
    htr_correction_auto: bool = False
    # gemma3:12b (~8.1 GB, Q4, no thinking mode): fits a 16 GB card together
    # with its KV cache, unlike gpt-oss:20b (13.8 GB MXFP4), which fails to
    # allocate and — being a reasoning model — spends the answer budget on
    #  reasoning tokens that the line-only sanitiser then rejects.
    htr_correction_model: str = "gemma3:12b"
    htr_correction_timeout_s: float = 90.0
    htr_correction_context_lines: int = 2
    htr_correction_max_lexicon: int = 200
    htr_correction_max_vocabulary: int = 200
    htr_correction_max_confusions: int = 25
    htr_correction_max_examples: int = 5

    # Vocabulary (dictionary) check of the recognized words: words missing from
    # the dictionary are highlighted in the UI so the user fixes them. The
    # artifact is built once by scripts/build_htr_lexicon.py; when the file is
    # missing the check is simply unavailable (no word is marked).
    htr_lexicon_enabled: bool = True
    htr_lexicon_path: str = "htr_storage/lexicon/ru_lexicon.bloom"

    # HTR fine-tuning (recognition training of per-author models)
    htr_epochs: int = 10
    htr_min_epochs: int = 0
    # 8 is kraken's own default and can fill a 16 GB card together with the
    # static-width compile; 4 leaves headroom and is plenty for small corpora.
    htr_batch_size: int = 4
    # A/B on the development corpus (87 lines, 8-line holdout) with the
    # auxiliary NRTR head disabled and BatchNorm statistics frozen:
    #   CER 0.212 base -> 0.160 at 1e-5 -> 0.110 at 1e-4
    # (the earlier "1e-4 destroyed accuracy" run was the random NRTR head plus
    # drifting BatchNorm statistics, not the learning rate). kraken's own
    # default for ppocrv6 medium is 5e-4, which needs far more data.
    htr_learning_rate: float = 0.0001
    htr_validation_split: float = 0.1
    htr_random_seed: int = 42
    htr_confidence_warning_threshold: float = 0.90
    htr_confidence_critical_threshold: float = 0.70
    # Fine-tuning starts only once the confirmed corpus reaches this size.
    # One page is usually ~30 lines, so the default waits for a second page.
    htr_min_training_lines: int = 50
    # Optional second threshold; 0 disables it (lines alone decide).
    htr_min_training_words: int = 0
    # Backend knobs handed to the trainer (kraken reads these keys).
    htr_training_resize: str = "union"          # fail | union | new
    # The recognizer stores NFC text, while the default model's codec uses
    # decomposed sequences ('и' + U+0306 for 'й'). Training must therefore be
    # normalized to NFD, otherwise 'й'/'ӗ' look like brand-new code points and
    # kraken resizes the output layer with random weights.
    htr_training_normalization: str = "NFD"     # NFD | NFKD | NFC | NFKC | none
    htr_training_height: int = 96               # informational: kraken uses the checkpoint's height
    # Force a different line height than the checkpoint declares (0 = keep it).
    # The pretrained weights of this project expect 96, so 128 trains in a
    # different scale; use it only for a deliberate A/B.
    htr_training_height_override: int = 0
    htr_training_max_width: int = 2560          # max line width after height normalization
    htr_training_variant: str = "medium"        # ppocrv6: tiny | small | medium
    htr_training_precision: str = "32-true"
    htr_training_num_workers: int = 0           # 0 keeps dataloading in-process
    htr_training_augment: bool = True
    htr_training_schedule: str = "cosine"
    # a short linear warmup stabilises the first steps on a tiny corpus
    htr_training_warmup: int = 10
    # Keep the pretrained backbone (everything but the codec projection) frozen
    # for the first steps, so a freshly resized output layer cannot drag the
    # learned features away (catastrophic forgetting on a small corpus).
    # Value is in optimizer steps, not samples — that is what kraken's callback
    # actually compares against (`trainer.global_step`).
    #   -1 = auto: the first epoch   0 = off   N > 0 = N steps
    htr_training_freeze_backbone: int = -1
    # Keep BatchNorm statistics frozen during the fine-tune. Batches here are 4
    # line crops, which is far too little to re-estimate 61 pretrained layers.
    htr_training_freeze_bn: bool = True
    # kraken's ppocrv6 recipe adds an auxiliary NRTR decoder loss to the CTC one
    # (loss = ctc + nrtr). The exported inference checkpoint we fine-tune has no
    # such head, so kraken builds it from random weights and the backbone starts
    # serving a random objective as soon as it is unfrozen. Leave it off unless
    # the base model really ships an NRTR head.
    htr_training_aux_nrtr: bool = False
    # How the training crops are declared to kraken. 'baselines' matches both
    # the dewarped crops (infrastructure/kraken/lines.py) and the base model;
    # 'bbox' forces the old box behaviour.
    htr_training_linetype: str = "baselines"
    htr_training_weight_decay: float = 0.01
    # torch.compile (inductor) codegen can take many minutes on a small corpus
    # and then dominates the run; eager mode is faster overall here. Enable it
    # only for large training sets.
    htr_training_compile: bool = False
    # TF32 matmuls on Tensor Core GPUs (meaningless/ignored on CPU).
    htr_training_matmul_precision: str = "high"

    class Config:
        env_file = ".env"


settings = Settings()

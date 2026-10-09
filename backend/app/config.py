import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Live Journal AI"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/life_logs"
    ollama_url: str = "http://localhost:11434"
    llama_model_path: str = "ai_models/Qwen3.5-9B.Q4_K_M.gguf"
    llama_n_ctx: int = 4096
    llama_n_threads: int = 8
    port: int = 8000

    # LLM that serves the chat, diary titles and knowledge extraction.
    # HTR line correction is a different job with its own model
    # (htr_correction_model below): it proposes per-line fixes and was tuned for
    # that, so it deliberately stays on a smaller non-reasoning model.
    llm_model: str = "qwen3:14b"
    # qwen3 is a reasoning model. Left on, it spends the answer budget on
    #  reasoning tokens: a page takes minutes longer and the strict JSON of
    # the extraction schemas comes back wrapped in prose. Structured calls
    # always disable it; the chat may keep it on.
    llm_thinking_enabled: bool = True
    # Loading a 14b model into VRAM at startup makes the first request fast at
    # the price of a slow boot; off by default on developer machines.
    llm_warmup_enabled: bool = False

    # Embeddings: deduplication of extracted knowledge and RAG over the archive.
    # The weights are a Hugging Face SentenceTransformer, not an Ollama model:
    # google/embeddinggemma-2 (740m: a 270m text backbone plus loadable vision
    # and audio encoders) returns 768-dimensional vectors. Loading is in-process,
    # so an embedding does not compete with the chat model for Ollama's queue.
    embedding_model: str = "google/embeddinggemma-2"
    embedding_dim: int = 768
    #: where the downloaded weights are cached; kept inside the project so the
    #: archive and its model stay together (htr_storage/ is gitignored)
    embedding_model_dir: str = "htr_storage/embedding_models"
    #: 'auto' takes CUDA when torch sees a GPU (the chat model shares it), else CPU
    embedding_device: str = "auto"
    #: Task instruction prefix for *symmetric* similarity (deduplication).
    #: Measured on 20 related + 20 unrelated Russian pairs (see the spec, 4.5):
    #: the model card's Clustering prompt is the worst choice here — unrelated
    #: phrases score 0.922 on average and the space collapses into a narrow cone
    #: (anisotropy 0.956), which is what makes "купить молоко" look like
    #: "похудеть к лету". SentenceSimilarity is the same symmetric task but keeps
    #: unrelated pairs at 0.814 (anisotropy 0.900) and ranks best (AUC 0.983).
    #: RAG passes its own asymmetric prompts: 'SearchQuery' vs 'Document'.
    embedding_default_prompt: str = "SentenceSimilarity"
    #: Load only the 270m text encoder: the vision and audio towers of this
    #: multimodal checkpoint are dead weight for the archive (memory 1.5 GB ->
    #: ~0.5 GB). Measured: identical vectors, see the spec 4.5.
    embedding_text_only: bool = True
    #: how many texts one forward pass sees
    embedding_batch_size: int = 32
    #: the checkpoint reports a nonsensical max_seq_length; the model card says
    #: 8192 tokens, and a sane cap keeps a long chunk from allocating wildly
    embedding_max_seq_length: int = 8192
    #: how many embedded strings one process keeps in memory — re-extracting a
    #: page asks for the same titles again and again
    embedding_cache_size: int = 1000
    #: load the weights at startup (~0.5 GB with embedding_text_only) instead of
    #: on the first request
    embedding_warmup_enabled: bool = False

    # Knowledge extraction pipeline. `sync` runs it inside the HTTP request
    # (the development default: no worker to babysit); `async` hands it to
    # Celery and lets the UI poll the source status.
    knowledge_processing_mode: str = "sync"
    # Deduplication. Measured with the default prompt (SentenceSimilarity) on 20
    # related + 20 unrelated Russian pairs: related 0.961±0.020 (min 0.922),
    # unrelated 0.814±0.039. No absolute threshold separates them cleanly — the
    # best one costs ~2 % errors and two *different* birthdays of the same child
    # still score 0.995 — so the embedding only *retrieves* candidates and the
    # LLM resolver decides. This floor is how far a candidate may fall before it
    # is not even shown to the model; it sits just above the unrelated mean, so
    # it filters noise without dropping a possible duplicate.
    knowledge_dedup_candidates: int = 5
    knowledge_dedup_min_similarity: float = 0.88

    # RAG in the chat.
    rag_enabled_by_default: bool = True
    rag_top_k: int = 4
    rag_chunk_max_chars: int = 1200
    rag_chunk_overlap_chars: int = 200

    # A confirmed manuscript page is extracted with the tail of the previous
    # page (and the head of the next one) as context: a sentence can start on
    # one page and end on another.
    knowledge_neighbor_context_chars: int = 400

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
    # Weight of the character language model. Re-tuned on *strict* ground truth
    # (only lines the user actually corrected) over honest acoustic folds — each
    # page recognized by the model that existed before its confirmation:
    #
    #   acoustic group   a=0.0        a=0.1 (old)   a=0.2 (new)
    #   weak  (v2)       .1047/.3342  .0887/.2722   .0807/.2488
    #   medium(v7-v9)    .0549/.2167  .0493/.1920   .0460/.1760
    #   strong(v10-v12)  .0324/.1484  .0300/.1342   .0240/.1053
    #
    # 0.2 wins in every group — the earlier 0.1 came from a *mixed* reference
    # (uncorrected lines compared against the previous model's own output), which
    # systematically punishes the LM for disagreeing with it. Note that the
    # measurement used a language model that includes the Leipzig corpus; if the
    # corpus changes, re-run scripts/measure_htr_decoders.py --only-corrected.
    htr_beam_alpha: float = 0.2
    htr_beam_beta: float = 0.0
    htr_beam_word_bonus: float = 0.8
    # Beam decoding needs a language model with real running text behind it;
    # below this many characters of running text (author pages plus --extra-text,
    # *not* bare word forms) the model cannot judge word boundaries and greedy
    # decoding is used instead.
    htr_lm_min_text_chars: int = 1000
    # Second pass: a word-level KenLM model re-ranks the beam's N best
    # hypotheses. Empty path (or a missing kenlm) disables it, and then decoding
    # is single-pass exactly as before. The weight stays small on purpose: the
    # word model is trained on general news text and pulls towards frequent
    # words, so it may only overrule an unsure first pass (see the guard below).
    # Second pass, measured honestly on chronological folds (8 pages, 239
    # lines): WER 0.2101 -> 0.2082 with the weight chosen on the other pages,
    # 0.2032 with the best weight chosen on all of them. That is +1..3 %, not
    # the +30 % an earlier run suggested: that run scored with a word model
    # that had the author's own pages baked in. When measuring this feature the
    # two components MUST be pure — a KenLM interpolated model is one merged
    # model, its halves cannot be un-mixed by re-weighting at query time.
    # Measured on strong acoustic folds and on the unseen pages 23/24: the word
    # model changes 0.05 % of characters there and *slightly worsens* WER
    # (0.0763 -> 0.0789 on v13). It stays implemented and tested, but off by
    # default; point this at the model to enable the second pass (worth
    # re-measuring once the author has much more confirmed text).
    htr_word_lm_path: str = ""
    #: the word score is an average per word, the beam score a total over ~40
    #: characters — 0.5 makes the two comparable; larger values let the word
    #: model dominate and measured worse once the leak was removed
    htr_rescore_weight: float = 0.5
    htr_rescore_n: int = 10
    #: the guard (allow a re-ranking only for an unsure top-1) is a wash on the
    #: honest model: the guarded optimum is 0.2044 against 0.2032 unguarded, and
    #: the leave-one-page-out selection picks unguarded on most pages. Kept as
    #: an option, off by default.
    htr_rescore_guard: bool = False
    htr_rescore_min_mean_acoustic: float = -0.30
    #: How many alternative readings to keep per recognized word for the editor
    #: (0 = keep none), and — because it sets how many hypotheses the beam is
    #: asked for — how many words get any alternative at all. Measured on pages
    #: 23/24 (382 words, v13): 5 hypotheses cover 42 % of words, 10 cover 56 %,
    #: 16 cover 65 %. Confidence does not explain the rest (mean confidence of
    #: words with variants 0.9885 against 0.9867 without): a word has no
    #: alternatives when *every* hypothesis spells it the same way, which is
    #: what happens when the beam spent its diversity elsewhere in the line.
    htr_word_alternatives: int = 10

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
    # Additionally group the baselines by the *physical rows* found in the ink
    # (the horizontal profile of the page): a logbook whose words and numbers
    # stand far apart arrives as several baselines per row. The geometric merge
    # above runs first, so the ink can only add merges, never remove one.
    htr_segmentation_rows: bool = True
    # Where a row band starts on the smoothed ink profile, as a fraction of its
    # maximum. Higher = narrower bands (tightly spaced diary rows must not be
    # glued), lower = wider (very sparse rows).
    htr_segmentation_rows_threshold: float = 0.4
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
    # Connecting to Ollama must fail fast: a stopped Ollama (or a stale Windows
    # host address) drops the connection silently instead of refusing it, so
    # with one shared timeout every dead attempt cost the full 90 s of
    # generation budget and a page took three minutes to come back empty.
    htr_correction_connect_timeout_s: float = 5.0
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
    # 0 disables the threshold: every confirmed page is the user's own decision
    # to include the material, so any non-empty corpus is trained on. Set it
    # (one page is usually ~30 lines, so 50 waits for a second page) to keep the
    # old "not enough data yet" gate.
    htr_min_training_lines: int = 0
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
    # Rebuild the author's character language model during a training run: it is
    # built from the same confirmed pages as the dataset, so a page returned to
    # editing has to disappear from it. Building reads tens of millions of
    # characters (tens of seconds), which is why it happens here and not on every
    # corpus change; failures are logged and never fail the training itself.
    htr_training_rebuild_lm: bool = True
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

# Hugging Face keeps both the downloaded weights and its transfer cache under
# $HF_HOME. Point it inside the project for two reasons: the archive and the
# model that reads it stay together (htr_storage/ is gitignored), and a machine
# whose ~/.cache is not writable (a sandbox, a shared box) still works.
# huggingface_hub resolves this at import time, so it must be set before
# anything imports transformers/sentence-transformers — config is the first
# project module every entry point loads, which is why it happens here.
os.environ.setdefault("HF_HOME", str(Path(settings.embedding_model_dir).resolve()))

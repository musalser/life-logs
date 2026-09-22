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

    # HTR recognition (inference)
    # Device string understood by kraken/lightning: 'cpu', 'cuda:0', 'auto'.
    htr_recognition_device: str = "cpu"
    htr_recognition_batch_size: int = 8
    htr_recognition_padding: int = 16
    htr_recognition_text_direction: str = "horizontal-lr"
    # 0 keeps line extraction inside the request process; >0 spawns workers,
    # which needs a working multiprocessing start method (unreliable on Windows).
    htr_recognition_num_line_workers: int = 0
    # Segmentation of the page into text lines (classical kraken segmenter).
    htr_segmentation_maxcolseps: int = 2
    htr_segmentation_no_hlines: bool = True

    # HTR fine-tuning (recognition training of per-author models)
    htr_epochs: int = 10
    htr_min_epochs: int = 0
    # 8 is kraken's own default and can fill a 16 GB card together with the
    # static-width compile; 4 leaves headroom and is plenty for small corpora.
    htr_batch_size: int = 4
    # 1e-5 keeps a pretrained model from forgetting it while adapting to a
    # small corpus (1e-4 destroyed accuracy on a 59-line first run)
    htr_learning_rate: float = 0.00001
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
    htr_training_height: int = 96               # line height (overridden by the loaded model)
    htr_training_max_width: int = 2560          # max line width after height normalization
    htr_training_variant: str = "medium"        # ppocrv6: tiny | small | medium
    htr_training_precision: str = "32-true"
    htr_training_num_workers: int = 0           # 0 keeps dataloading in-process
    htr_training_augment: bool = True
    htr_training_schedule: str = "cosine"
    # a short linear warmup stabilises the first steps on a tiny corpus
    htr_training_warmup: int = 10
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

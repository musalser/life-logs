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
    htr_default_model_path: str = "htr_storage/models/default/default_model.mlmodel"
    htr_device: str = "cuda:0"
    htr_epochs: int = 50
    htr_batch_size: int = 8
    htr_learning_rate: float = 0.0001
    htr_validation_split: float = 0.1
    htr_random_seed: int = 42
    htr_confidence_warning_threshold: float = 0.90
    htr_confidence_critical_threshold: float = 0.70
    htr_min_training_samples: int = 1

    class Config:
        env_file = ".env"


settings = Settings()

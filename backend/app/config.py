from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Live Journal AI"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/life_logs"
    ollama_url: str = "http://localhost:11434"
    llama_model_path: str = "ai_models/Qwen3.5-9B.Q4_K_M.gguf"
    llama_n_ctx: int = 4096
    llama_n_threads: int = 8
    port: int = 8000

    # Auth
    jwt_secret_key: str = "change-me-in-env"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    auth_username: str = "admin"
    auth_password: str = "admin"

    # Celery / Redis
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    class Config:
        env_file = ".env"


settings = Settings()

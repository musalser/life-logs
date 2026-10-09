from celery import Celery
from .config import settings

celery = Celery(
    "life_logs",
    # `include` is what makes the tasks visible to a worker: without it the
    # worker only imports app.celery_app, registers nothing and answers every
    # dispatch with NotRegistered ('app.tasks.extract_knowledge').
    include=["app.tasks"],
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery.conf.task_routes = {"app.tasks.*": {"queue": "default"}}
celery.conf.result_expires = 3600
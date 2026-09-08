from celery import Celery
from .config import settings

celery = Celery(
    "life_logs",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery.conf.task_routes = {"app.tasks.*": {"queue": "default"}}
celery.conf.result_expires = 3600
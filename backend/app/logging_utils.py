import logging
import time
from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request


request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx_var.get()
        return True


def configure_logging() -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - [request_id=%(request_id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    request_id_filter = RequestIdFilter()
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ]

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(request_id_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)


async def log_requests(request: Request, call_next):
    logger = logging.getLogger("app.request")
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    token = request_id_ctx_var.set(request_id)
    started_at = time.perf_counter()

    logger.info("Started %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "Failed %s %s in %.2f ms",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    else:
        duration_ms = (time.perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "Completed %s %s with %s in %.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    finally:
        request_id_ctx_var.reset(token)
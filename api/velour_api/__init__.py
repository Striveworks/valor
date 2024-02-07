import importlib.metadata
import os

import structlog

try:
    logging_level = int(os.getenv("LOGGING_LEVEL", 20))
except (TypeError, ValueError):
    logging_level = 20


structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging_level),
)

logger = structlog.get_logger()

try:
    __version__ = importlib.metadata.version("velour-api")
except importlib.metadata.PackageNotFoundError:
    __version__ = ""

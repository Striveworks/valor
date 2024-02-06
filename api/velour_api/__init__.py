import importlib.metadata
import logging
import os

import structlog

try:
    logging_level = getattr(
        logging, os.environ.get("LOGGING_LEVEL", "INFO").upper()
    )
except AttributeError:
    logging_level = getattr(logging, "INFO")


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

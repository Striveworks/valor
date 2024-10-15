import importlib.metadata
import os
import warnings

import structlog

try:
    logging_level = int(os.getenv("LOGGING_LEVEL", 20))
except (TypeError, ValueError):
    logging_level = 20

warnings.filterwarnings(
    "ignore",
    message=r'.*Field "model_.*" has conflict with protected namespace "model_".*',
)


def status_endpoint_filter(
    logger,
    method_name,
    event_dict,
    ignore_paths=frozenset(["/health", "/ready"]),
):
    if (
        event_dict.get("path", "") in ignore_paths
        and event_dict.get("status", 0) == 200
    ):
        raise structlog.DropEvent
    return event_dict


structlog.configure(
    processors=[
        status_endpoint_filter,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging_level),
)

logger = structlog.get_logger()

try:
    __version__ = importlib.metadata.version("valor-api")
except importlib.metadata.PackageNotFoundError:
    __version__ = ""

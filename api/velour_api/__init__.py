import importlib.metadata
import logging.config

from . import settings

logging.config.dictConfig(settings.LogConfig().model_dump())
logger = logging.getLogger("velour-backend")

try:
    __version__ = importlib.metadata.version("velour-api")
except importlib.metadata.PackageNotFoundError:
    __version__ = ""

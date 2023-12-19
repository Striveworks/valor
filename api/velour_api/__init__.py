import importlib.metadata
import logging.config

from . import settings

__version__ = importlib.metadata.version("velour_api")

logging.config.dictConfig(settings.LogConfig().model_dump())
logger = logging.getLogger("velour-backend")

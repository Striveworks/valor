import logging.config

from . import settings

# __all__ = ["logger"]

logging.config.dictConfig(settings.LogConfig().dict())
logger = logging.getLogger("velour-backend")

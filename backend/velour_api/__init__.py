import logging.config

from . import settings

logging.config.dictConfig(settings.LogConfig().dict())
logger = logging.getLogger("velour-backend")

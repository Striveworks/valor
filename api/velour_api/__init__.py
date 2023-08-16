import logging.config

from . import settings

logging.config.dictConfig(settings.LogConfig().model_dump())
logger = logging.getLogger("velour-backend")

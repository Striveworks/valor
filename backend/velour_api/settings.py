from pydantic import BaseSettings


class LogConfig(BaseSettings):
    """Logging configuration to be set for the server

    (taken from https://stackoverflow.com/a/67937084 but using
    BaseSettings)
    """

    LOGGER_NAME: str = "velour-backend"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    # Logging config
    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers = {
        LOGGER_NAME: {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    }


class AuthConfig(BaseSettings):
    domain: str = None
    audience: str = None
    algorithms: str = None

    class Config:
        env_file = ".env.auth"

    @property
    def no_auth(self) -> bool:
        return all([not v for v in self.dict().values()])

    @property
    def jwks_url(self) -> str:
        if self.domain:
            return f"https://{self.domain}/.well-known/jwks.json"
        return None

    @property
    def issuer(self) -> str:
        return f"https://{self.domain}/"


auth_settings = AuthConfig()

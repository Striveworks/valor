from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class LogConfig(BaseSettings):
    """Logging configuration to be set for the server

    (taken from https://stackoverflow.com/a/67937084 but using
    BaseSettings)
    """

    LOGGER_NAME: str = "velour-backend"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    # Logging config
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers: dict = {
        LOGGER_NAME: {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    }


class AuthConfig(BaseSettings):
    SECRET_KEY: str | None = None
    ALGORITHM: str | None = "HS256"
    USERNAME: str | None = None
    PASSWORD: str | None = None
    model_config = ConfigDict(env_file=".env.auth")

    @property
    def no_auth(self) -> bool:
        return not bool(self.SECRET_KEY)

    @model_validator(mode="after")
    def check_all_fields(self):
        """Makes sure that either all of SECRET_KEY, USERNAME, and PASSWORD
        are set or none of them are.
        """
        bools = [
            bool(self.SECRET_KEY),
            bool(self.USERNAME),
            bool(self.PASSWORD),
        ]
        if any(bools) and not all(bools):
            raise ValueError(
                "Either all of SECRET_KEY, USERNAME, and PASSWORD must be set or none of them must be set."
            )

        return self


auth_settings = AuthConfig()

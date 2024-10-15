from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class AuthConfig(BaseSettings):
    SECRET_KEY: str | None = None
    ALGORITHM: str | None = "HS256"
    USERNAME: str | None = None
    PASSWORD: str | None = None
    model_config = ConfigDict(env_file=".env.auth", env_prefix="VALOR_")  # type: ignore - pydantic error; type "ConfigDict" cannot be assigned to declared type "SettingsConfigDict"

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

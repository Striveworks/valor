from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from valor_api.schemas.metadata import Metadata
from valor_api.schemas.validators import validate_string_identifier


class Datum(BaseModel):
    uid: str
    metadata: Metadata | None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.uid)
        return self


class Dataset(BaseModel):
    name: str
    metadata: Metadata | None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.name)
        return self


class Model(BaseModel):
    name: str
    metadata: Metadata | None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.name)
        return self

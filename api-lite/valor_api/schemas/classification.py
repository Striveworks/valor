from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from valor_api.schemas.metadata import Metadata
from valor_api.schemas.validators import validate_string_identifier


class Classification(BaseModel):
    uid: str
    groundtruth: str
    predictions: list[str]
    scores: list[float]
    metadata: Metadata | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        validate_string_identifier(self.uid)
        validate_string_identifier(self.groundtruth)
        for prediction in self.predictions:
            validate_string_identifier(prediction)
        return self

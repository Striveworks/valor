from pydantic import BaseModel, ConfigDict, Field, field_validator

from velour_api import schemas
from velour_api.enums import AnnotationType, TaskType


def _validate_comparison_operators(op: str):
    if op not in [">", "<", ">=", "<=", "==", "!="]:
        raise ValueError(f"`{op}` is not a valid comparison operator")
    return op


class StringFilter(BaseModel):
    value: str
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        if op is not None:
            _validate_comparison_operators(op)
        return op

    model_config = ConfigDict(extra="forbid")


class NumericFilter(BaseModel):
    value: float
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        if op is not None:
            _validate_comparison_operators(op)
        return op

    model_config = ConfigDict(extra="forbid")


class GeospatialFilter(BaseModel):
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        if op is not None:
            _validate_comparison_operators(op)
        return op

    model_config = ConfigDict(extra="forbid")


class GeometricFilter(BaseModel):
    area: NumericFilter | None = None
    height: NumericFilter | None = None
    width: NumericFilter | None = None


class MetaDatumFilter(BaseModel):
    """target.filter(metadatum, operator) ==> target > metadatum"""

    key: str
    value: NumericFilter | StringFilter | GeospatialFilter
    model_config = ConfigDict(extra="forbid")


class DatasetFilter(BaseModel):
    names: list[str] = Field(default_factory=list)
    metadata: list[MetaDatumFilter] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ModelFilter(BaseModel):
    names: list[str] = Field(default_factory=list)
    metadata: list[MetaDatumFilter] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class DatumFilter(BaseModel):
    uids: list[str] = Field(default_factory=list)
    metadata: list[MetaDatumFilter] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AnnotationFilter(BaseModel):

    # filter by type
    task_types: list[TaskType] = Field(default_factory=list)
    annotation_types: list[AnnotationType] = Field(default_factory=list)

    # filter by attributes
    geometry: GeometricFilter | None = None

    # filter by metadata
    metadata: list[MetaDatumFilter] = Field(default_factory=list)

    # toggle
    allow_conversion: bool = False

    model_config = ConfigDict(extra="forbid")


class LabelFilter(BaseModel):
    labels: list[schemas.Label] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class GroundTruthFilter(BaseModel):
    labels: LabelFilter | None = None
    model_config = ConfigDict(extra="forbid")


class PredictionFilter(BaseModel):
    labels: LabelFilter | None = None
    score: NumericFilter | None = None
    model_config = ConfigDict(extra="forbid")


class Filter(BaseModel):
    # filters
    datasets: DatasetFilter | None = None
    models: ModelFilter | None = None
    datums: DatumFilter | None = None
    annotations: AnnotationFilter | None = None
    groundtruths: GroundTruthFilter | None = None
    predictions: PredictionFilter | None = None
    labels: LabelFilter | None = None

    model_config = ConfigDict(extra="forbid")

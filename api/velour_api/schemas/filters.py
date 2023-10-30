from pydantic import BaseModel, ConfigDict, Field, field_validator

from velour_api import schemas
from velour_api.enums import AnnotationType, TaskType


class StringFilter(BaseModel):
    value: str
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        allowed_operators = ["==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")


class NumericFilter(BaseModel):
    value: float
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")


class GeospatialFilter(BaseModel):
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")


class MetadatumFilter(BaseModel):
    """target.filter(metadatum, operator) ==> target > metadatum"""

    key: str
    comparison: NumericFilter | StringFilter | GeospatialFilter
    model_config = ConfigDict(extra="forbid")


class DatasetFilter(BaseModel):
    ids: list[int] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    metadata: list[MetadatumFilter] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ModelFilter(BaseModel):
    ids: list[int] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    metadata: list[MetadatumFilter] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class DatumFilter(BaseModel):
    ids: list[int] = Field(default_factory=list)
    uids: list[str] = Field(default_factory=list)
    metadata: list[MetadatumFilter] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class GeometricFilter(BaseModel):
    annotation_type: AnnotationType
    area: list[NumericFilter] = Field(default_factory=list)

    @field_validator("annotation_type")
    @classmethod
    def validate_annotation_type(cls, annotation_type):
        if annotation_type not in [
            AnnotationType.BOX,
            AnnotationType.POLYGON,
            AnnotationType.MULTIPOLYGON,
            AnnotationType.RASTER,
        ]:
            raise ValueError(
                f"Expected geometric annotation type, got `{annotation_type}`."
            )
        return annotation_type


class AnnotationFilter(BaseModel):
    task_types: list[TaskType] = Field(default_factory=list)
    geometries: list[GeometricFilter] = Field(default_factory=list)
    metadata: list[MetadatumFilter] = Field(default_factory=list)

    # TODO
    allow_conversion: bool = False

    model_config = ConfigDict(extra="forbid")


class LabelFilter(BaseModel):
    ids: list[int] = Field(default_factory=list)
    labels: list[schemas.Label] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class PredictionFilter(BaseModel):
    scores: list[NumericFilter] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class Filter(BaseModel):
    datasets: DatasetFilter = Field(default=DatasetFilter())
    models: ModelFilter = Field(default=ModelFilter())
    datums: DatumFilter = Field(default=DatumFilter())
    annotations: AnnotationFilter = Field(default=AnnotationFilter())
    predictions: PredictionFilter = Field(default=PredictionFilter())
    labels: LabelFilter = Field(default=LabelFilter())

    # pydantic settings
    model_config = ConfigDict(extra="forbid")

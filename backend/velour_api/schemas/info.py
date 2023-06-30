from pydantic import BaseModel, validator

from velour_api.enums import AnnotationType, DatumTypes
from velour_api.schemas.label import Label
from velour_api.schemas.metadata import MetaDatum


def _validate_href(v: str | None):
    if v is None:
        return v
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")
    return v


class LabelDistribution(BaseModel):
    label: Label
    count: int


class ScoredLabelDistribution(BaseModel):
    label: Label
    count: int
    scores: list[float]


class AnnotationDistribution:
    annotation_type: AnnotationType
    count: int


class DatasetMetadata(BaseModel):
    href: str = None
    description: str = None
    type: DatumTypes
    label: Label
    annotations: list[AnnotationType]
    labels: list[LabelDistribution]
    associated_models: list[str]
    metadata: list[MetaDatum]
    
    @validator("href")
    def validate_href(cls, v):
        return _validate_href(v)


class ModelMetadata(BaseModel):
    href: str = None
    description: str = None
    type: DatumTypes
    annotations: list[AnnotationType]
    labels: list[ScoredLabelDistribution]
    associated_datasets: list[str]
    metadata: list[MetaDatum]
    
    @validator("href")
    def validate_href(cls, v):
        return _validate_href(v)

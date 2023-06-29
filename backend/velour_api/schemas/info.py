from pydantic import BaseModel

from velour_api.enums import AnnotationType
from velour_api.schemas.core import Dataset, Model
from velour_api.schemas.label import Label, ScoredLabel


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


class DatasetInfo(BaseModel):
    annotations: list[AnnotationType]
    labels: list[LabelDistribution]
    associated_models: list[Model]


class ModelInfo(BaseModel):
    annotations: list[AnnotationType]
    labels: list[ScoredLabelDistribution]
    associated_datasets: list[Dataset]

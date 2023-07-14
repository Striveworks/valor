from pydantic import BaseModel, validator

from velour_api.enums import AnnotationType
from velour_api.schemas.label import Label


class LabelDistribution(BaseModel):
    label: Label
    count: int


class ScoredLabelDistribution(BaseModel):
    label: Label
    count: int
    scores: list[float]


class AnnotationDistribution(BaseModel):
    annotation_type: AnnotationType
    count: int

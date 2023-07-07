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


class AnnotationDistribution(BaseModel):
    annotation_type: AnnotationType
    count: int

from pydantic import BaseModel, Field

from velour_api import schemas
from velour_api.enums import AnnotationType, TaskType


class LabelDistribution(BaseModel):
    label: schemas.Label
    count: int


class ScoredLabelDistribution(BaseModel):
    label: schemas.Label
    count: int
    scores: list[float]


class AnnotationDistribution(BaseModel):
    annotation_type: AnnotationType
    count: int


class Filter(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    datum_uids: list[str] = Field(default_factory=list)
    task_types: list[TaskType] = Field(default_factory=list)
    annotation_types: list[AnnotationType] = Field(default_factory=list)
    labels: list[schemas.Label] = Field(default_factory=list)
    label_keys: list[str] = Field(default_factory=list)
    metadata: list[schemas.MetaDatum] = Field(default_factory=list)

    allow_dataset_metadata: bool = True
    allow_model_metadata: bool = True
    allow_datum_metadata: bool = True
    allow_annotation_metadata: bool = True
    allow_predictions: bool = True
    allow_groundtruths: bool = True

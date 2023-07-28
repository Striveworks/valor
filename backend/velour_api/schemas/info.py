from pydantic import BaseModel, validator

from velour_api import schemas
from velour_api.enums import Table, AnnotationType, TaskType


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
    dataset_names: list[str] = []
    model_names: list[str] = []
    datum_uids: list[str] = []
    task_types: list[TaskType] = []
    annotation_types: list[AnnotationType] = []
    labels: list[schemas.Label] = []
    label_keys: list[str] = []
    metadata: list[schemas.MetaDatum] = []

    allow_dataset_metadata: bool = True
    allow_model_metadata: bool = True
    allow_datum_metadata: bool = True
    allow_annotation_metadata: bool = True 
    allow_predictions: bool = True
    allow_groundtruths: bool = True
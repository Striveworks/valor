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
    filter_by_dataset_names: list[str] = []
    filter_by_model_names: list[str] | None = []
    filter_by_datum_uids: list[str] = []
    filter_by_task_types: list[TaskType] = []
    filter_by_annotation_types: list[AnnotationType] = []
    filter_by_labels: list[schemas.Label] = []
    filter_by_label_keys: list[str] = []
    filter_by_metadata: list[schemas.MetaDatum] = []
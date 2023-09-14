from pydantic import BaseModel, ConfigDict, Field

from velour_api.enums import AnnotationType, TaskType
from velour_api.schemas.core import Label
from velour_api.schemas.metadata import Metadatum


class Filter(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    datum_uids: list[str] = Field(default_factory=list)
    task_types: list[TaskType] = Field(default_factory=list)
    annotation_types: list[AnnotationType] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)
    label_keys: list[str] = Field(default_factory=list)
    metadata: list[Metadatum] = Field(default_factory=list)

    allow_dataset_metadata: bool = True
    allow_model_metadata: bool = True
    allow_datum_metadata: bool = True
    allow_annotation_metadata: bool = True
    allow_predictions: bool = True
    allow_groundtruths: bool = True

    model_config = ConfigDict(extra="forbid")

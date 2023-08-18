from pydantic import BaseModel, ConfigDict, Field

from velour_api import schemas
from velour_api.enums import AnnotationType, TaskType


class DatasetFilter(BaseModel):
    names: list[str] = Field(default_factory=list)
    metadata: list[schemas.MetaDatum] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ModelFilter(BaseModel):
    names: list[str] = Field(default_factory=list)
    metadata: list[schemas.MetaDatum] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class DatumFilter(BaseModel):
    uids: list[str] = Field(default_factory=list)
    metadata: list[schemas.MetaDatum] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AnnotationFilter(BaseModel):

    # filter by type
    task_types: list[TaskType] = Field(default_factory=list)
    annotation_types: list[AnnotationType] = Field(default_factory=list)

    # filter by geometric area
    min_area: float | None = None
    max_area: float | None = None

    # filter by metadata
    metadata: list[schemas.MetaDatum] = Field(default_factory=list)

    # toggle
    allow_conversion: bool = False

    model_config = ConfigDict(extra="forbid")


class LabelFilter(BaseModel):
    labels: list[schemas.Label] = Field(default_factory=list)
    label_keys: list[str] = Field(default_factory=list)

    # toggle graph nodes
    include_predictions: bool = True
    include_groundtruths: bool = True

    model_config = ConfigDict(extra="forbid")


class Filter(BaseModel):

    datasets: DatasetFilter
    models: ModelFilter
    datums: DatumFilter
    annotations: AnnotationFilter
    labels: LabelFilter

    model_config = ConfigDict(extra="forbid")

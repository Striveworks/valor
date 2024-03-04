from pydantic import BaseModel

from valor_api.enums import TaskType

from .core import MetadataType
from .label import Label


class DatasetSummary(BaseModel):
    name: str
    num_datums: int
    num_annotations: int
    num_bounding_boxes: int
    num_polygons: int
    num_rasters: int
    task_types: list[TaskType]
    labels: list[Label]
    datum_metadata: list[MetadataType]
    annotation_metadata: list[MetadataType]

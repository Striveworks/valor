from pydantic import BaseModel

from velour_api.enums import TaskType

from .core import MetadataType
from .label import Label


class DatasetSummary(BaseModel):
    name: str
    num_datums: int
    num_groundtruth_annotations: int
    num_groundtruth_bounding_boxes: int
    num_groundtruth_polygons: int
    num_groundtruth_multipolygons: int
    num_groundtruth_rasters: int
    task_types: list[TaskType]
    labels: list[Label]
    datum_metadata: list[MetadataType]
    groundtruth_annotation_metadata: list[MetadataType]

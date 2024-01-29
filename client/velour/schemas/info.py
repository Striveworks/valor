from dataclasses import dataclass
from typing import List

from velour.enums import TaskType
from velour.schemas.core import Label
from velour.types import MetadataType


@dataclass
class DatasetSummary:
    """Dataclass for storing dataset summary information"""

    name: str
    num_datums: int
    num_annotations: int
    num_bounding_boxes: int
    num_polygons: int
    num_groundtruth_multipolygons: int
    num_rasters: int
    task_types: List[TaskType]
    labels: List[Label]
    datum_metadata: List[MetadataType]
    annotation_metadata: List[MetadataType]

    def __post_init__(self):
        for i, tt in enumerate(self.task_types):
            if isinstance(tt, str):
                self.task_types[i] = TaskType(tt)
        for i, label in enumerate(self.labels):
            if isinstance(label, dict):
                self.labels[i] = Label(**label)

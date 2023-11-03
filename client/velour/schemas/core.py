from dataclasses import dataclass, field
from typing import Dict

from velour.enums import TaskType
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.label import Label
from velour.schemas.metadata import _BaseMetadatum


@dataclass
class _BaseDatum:
    uid: str
    dataset: str = field(default="")
    metadata: list[_BaseMetadatum] = field(default_factory=list)


@dataclass
class _BaseAnnotation:
    task_type: TaskType
    labels: list[Label] = field(default_factory=list)
    metadata: list[_BaseMetadatum] = field(default_factory=list)
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None
    jsonb: Dict = None


@dataclass
class _BaseGroundTruth:
    datum: _BaseDatum
    annotations: list[_BaseAnnotation] = field(default_factory=list)


@dataclass
class _BasePrediction:
    datum: _BaseDatum
    model: str
    annotations: list[_BaseAnnotation] = field(default_factory=list)

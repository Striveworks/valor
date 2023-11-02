import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from velour.enums import TaskType
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.metadata import GeoJSON


def _validate_href(v: str):
    if not isinstance(v, str):
        raise TypeError("passed something other than 'str'")
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


@dataclass
class Label_:
    key: str
    value: str
    score: Union[float, None] = None


@dataclass
class Datum_:
    uid: str
    metadata: List[Metadatum_] = field(default_factory=list)
    id: Union[int, None] = None


@dataclass
class Annotation_:
    task_type: TaskType
    labels: List[Label_] = field(default_factory=list)
    metadata: List[Metadatum_] = field(default_factory=list)
    bounding_box: Union[BoundingBox, None] = None
    polygon: Union[Polygon, None] = None
    multipolygon: Union[MultiPolygon, None] = None
    raster: Union[Raster, None] = None
    jsonb: Union[Dict, None] = None


@dataclass
class GroundTruth_:
    datum: Datum_
    annotations: List[Annotation_] = field(default_factory=list)


@dataclass
class Prediction_:
    datum: Datum_
    annotations: List[Annotation_] = field(default_factory=list)
    model: str = field(default="")

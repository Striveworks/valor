from abc import ABC
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np


@dataclass
class Image:
    uid: str
    height: int
    width: int


@dataclass
class Label:
    key: str
    value: str

    def tuple(self) -> Tuple[str, str]:
        return (self.key, self.value)


@dataclass
class ScoredLabel:
    label: Label
    score: float


@dataclass
class Point:
    x: float
    y: float

    def resize(
        self, og_img_h: int, og_img_w: int, new_img_h: int, new_img_w: int
    ) -> "Point":
        h_factor, w_factor = new_img_h / og_img_h, new_img_w / og_img_w
        return Point(x=w_factor * self.x, y=h_factor * self.y)


@dataclass
class BoundingPolygon:
    """Class for representing a bounding region."""

    points: List[Point]

    def xy_list(self):
        return [(pt.x, pt.y) for pt in self.points]

    @property
    def xmin(self):
        return min(p.x for p in self.points)

    @property
    def ymin(self):
        return min(p.y for p in self.points)

    @property
    def xmax(self):
        return max(p.x for p in self.points)

    @property
    def ymax(self):
        return max(p.y for p in self.points)

    @classmethod
    def from_ymin_xmin_ymax_xmax(cls, ymin, xmin, ymax, xmax):
        return cls(
            points=[
                Point(xmin, ymin),
                Point(xmin, ymax),
                Point(xmax, ymax),
                Point(xmax, ymin),
            ]
        )


@dataclass
class GroundTruthDetection:
    boundary: BoundingPolygon
    labels: List[Label]
    image: Image


@dataclass
class PredictedDetection:
    boundary: BoundingPolygon
    scored_labels: List[ScoredLabel]
    image: Image

    @property
    def labels(self):
        return [sl.label for sl in self.scored_labels]


@dataclass
class PolygonWithHole:
    polygon: BoundingPolygon
    hole: BoundingPolygon = None


def _validate_mask(mask: np.ndarray):
    if mask.dtype != bool:
        raise ValueError(
            f"Expecting a binary mask (i.e. of dtype bool) but got dtype {mask.dtype}"
        )


@dataclass
class _GroundTruthSegmentation(ABC):
    shape: Union[List[PolygonWithHole], np.ndarray]
    labels: List[Label]
    image: Image
    _is_instance: bool

    def __post_init__(self):
        if self.__class__ == _GroundTruthSegmentation:
            raise TypeError("Cannot instantiate abstract class.")
        if isinstance(self.shape, np.ndarray):
            _validate_mask(self.shape)


@dataclass
class GroundTruthInstanceSegmentation(_GroundTruthSegmentation):
    _is_instance: bool = field(default=True, init=False)


@dataclass
class GroundTruthSemanticSegmentation(_GroundTruthSegmentation):
    _is_instance: bool = field(default=False, init=False)


@dataclass
class _PredictedSegmentation(ABC):
    mask: np.ndarray
    scored_labels: List[ScoredLabel]
    image: Image
    _is_instance: bool

    def __post_init__(self):
        if self.__class__ == _GroundTruthSegmentation:
            raise TypeError("Cannot instantiate abstract class.")
        _validate_mask(self.mask)


@dataclass
class PredictedInstanceSegmentation(_PredictedSegmentation):
    _is_instance: bool = field(default=True, init=False)


@dataclass
class PredictedSemanticSegmentation(_PredictedSegmentation):
    _is_instance: bool = field(default=False, init=False)


@dataclass
class GroundTruthImageClassification:
    image: Image
    labels: List[Label]


@dataclass
class PredictedImageClassification:
    image: Image
    scored_labels: List[ScoredLabel]

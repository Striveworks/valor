from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    labels: list[tuple[str, str]]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.scores) > 0 and len(self.labels) != len(self.scores):
            raise ValueError(
                "If scores are defined, there must be a 1:1 pairing with labels."
            )

    @property
    def extrema(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)


@dataclass
class Polygon:
    shape: ShapelyPolygon
    labels: list[tuple[str, str]]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.shape, ShapelyPolygon):
            raise TypeError("shape must be of type shapely.geometry.Polygon.")
        if len(self.scores) > 0 and len(self.labels) != len(self.scores):
            raise ValueError(
                "If scores are defined, there must be a 1:1 pairing with labels."
            )


@dataclass
class Bitmask:
    mask: NDArray[np.bool_]
    labels: list[tuple[str, str]]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.scores) > 0 and len(self.labels) != len(self.scores):
            raise ValueError(
                "If scores are defined, there must be a 1:1 pairing with labels."
            )

    def to_box(self) -> BoundingBox | None:

        if not self.mask.any():
            return None

        rows, cols = np.nonzero(self.mask)
        return BoundingBox(
            xmin=cols.min(),
            xmax=cols.max(),
            ymin=rows.min(),
            ymax=rows.max(),
            labels=self.labels,
            scores=self.scores,
        )


@dataclass
class Detection:
    uid: str
    groundtruths: list[BoundingBox] | list[Bitmask] | list[Polygon]
    predictions: list[BoundingBox] | list[Bitmask] | list[Polygon]

    def __post_init__(self):
        for prediction in self.predictions:
            if len(prediction.scores) != len(prediction.labels):
                raise ValueError(
                    "Predictions must provide a score for every label."
                )

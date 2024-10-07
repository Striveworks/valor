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
    labels: list[str]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.scores) == 0 and len(self.labels) != 1:
            raise ValueError(
                "If no scores are defined, then this is a ground truth and a single label requirement is enforced."
            )
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
    labels: list[str]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.shape, ShapelyPolygon):
            raise TypeError("shape must be of type shapely.geometry.Polygon.")

        if len(self.scores) == 0 and len(self.labels) != 1:
            raise ValueError(
                "If no scores are defined, then this is a ground truth and a single label requirement is enforced."
            )
        if len(self.scores) > 0 and len(self.labels) != len(self.scores):
            raise ValueError(
                "If scores are defined, there must be a 1:1 pairing with labels."
            )

    def to_box(self) -> BoundingBox | None:

        if self.shape.is_empty:
            return None

        xmin, ymin, xmax, ymax = self.shape.bounds

        return BoundingBox(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            labels=self.labels,
            scores=self.scores,
        )


@dataclass
class Bitmask:
    mask: NDArray[np.bool_]
    labels: list[str]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self):

        if (
            not isinstance(self.mask, np.ndarray)
            or self.mask.dtype != np.bool_
        ):
            raise ValueError(
                "Expected mask to be of type `NDArray[np.bool_]`."
            )

        if len(self.scores) == 0 and len(self.labels) != 1:
            raise ValueError(
                "If no scores are defined, then this is a ground truth and a single label requirement is enforced."
            )
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

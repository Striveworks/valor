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
                "Ground truths must be defined with no scores and a single label. If you meant to define a prediction, then please include one score for every label provided."
            )
        if len(self.scores) > 0 and len(self.labels) != len(self.scores):
            raise ValueError(
                "If scores are defined, there must be a 1:1 pairing with labels."
            )

    @property
    def extrema(self) -> tuple[float, float, float, float]:
        """
        Returns annotation extrema in the form (xmin, xmax, ymin, ymax).
        """
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def annotation(self) -> tuple[float, float, float, float]:
        """
        Returns the annotation's data representation.
        """
        return self.extrema


@dataclass
class Polygon:
    shape: ShapelyPolygon
    labels: list[str]
    scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.shape, ShapelyPolygon):
            raise TypeError("shape must be of type shapely.geometry.Polygon.")
        if self.shape.is_empty:
            raise ValueError("Polygon is empty.")

        if len(self.scores) == 0 and len(self.labels) != 1:
            raise ValueError(
                "Ground truths must be defined with no scores and a single label. If you meant to define a prediction, then please include one score for every label provided."
            )
        if len(self.scores) > 0 and len(self.labels) != len(self.scores):
            raise ValueError(
                "If scores are defined, there must be a 1:1 pairing with labels."
            )

    @property
    def extrema(self) -> tuple[float, float, float, float]:
        """
        Returns annotation extrema in the form (xmin, xmax, ymin, ymax).
        """
        xmin, ymin, xmax, ymax = self.shape.bounds
        return (xmin, xmax, ymin, ymax)

    @property
    def annotation(self) -> ShapelyPolygon:
        """
        Returns the annotation's data representation.
        """
        return self.shape


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
        elif not self.mask.any():
            raise ValueError("Mask does not define any object instances.")

        if len(self.scores) == 0 and len(self.labels) != 1:
            raise ValueError(
                "Ground truths must be defined with no scores and a single label. If you meant to define a prediction, then please include one score for every label provided."
            )
        if len(self.scores) > 0 and len(self.labels) != len(self.scores):
            raise ValueError(
                "If scores are defined, there must be a 1:1 pairing with labels."
            )

    @property
    def extrema(self) -> tuple[float, float, float, float]:
        """
        Returns annotation extrema in the form (xmin, xmax, ymin, ymax).
        """
        rows, cols = np.nonzero(self.mask)
        return (cols.min(), cols.max(), rows.min(), rows.max())

    @property
    def annotation(self) -> NDArray[np.bool_]:
        """
        Returns the annotation's data representation.
        """
        return self.mask


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

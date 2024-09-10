from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    labels: list[tuple[str, str]]
    scores: list[float] | None = None

    def __post_init__(self):
        if self.scores is not None:
            if len(self.labels) != len(self.scores):
                raise ValueError

    @property
    def extrema(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)


@dataclass
class Bitmask:
    mask: NDArray[np.bool_]
    labels: list[tuple[str, str]]
    scores: list[float] | None = None

    def __post_init__(self):
        if self.scores is not None:
            if len(self.labels) != len(self.scores):
                raise ValueError

    def to_box(self) -> BoundingBox:
        raise NotImplementedError


@dataclass
class Detection:
    uid: str
    groundtruths: list[BoundingBox]
    predictions: list[BoundingBox]

    def __post_init__(self):
        for prediction in self.predictions:
            if prediction.scores is None:
                raise ValueError

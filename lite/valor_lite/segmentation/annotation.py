from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Bitmask:
    mask: NDArray[np.bool_]
    label: tuple[str, str]


@dataclass
class Segmentation:
    uid: str
    groundtruths: list[Bitmask]
    predictions: list[Bitmask]
    scores: list[float]

    def __post_init__(self):
        if len(self.predictions) != len(self.scores):
            raise ValueError("There must be a score per prediction label.")

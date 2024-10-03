from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Bitmask:
    mask: NDArray[np.bool_]
    label: str


@dataclass
class WeightedMask:
    mask: NDArray[np.float64]
    label: str


@dataclass
class Segmentation:
    uid: str
    groundtruths: list[Bitmask]
    predictions: list[WeightedMask]
    shape: tuple[int, ...] = field(default_factory=lambda: (0, 0))
    size: int = field(default=0)

    def __post_init__(self):

        groundtruth_shape = {
            groundtruth.mask.shape for groundtruth in self.groundtruths
        }
        prediction_shape = {
            prediction.mask.shape for prediction in self.predictions
        }
        if (
            len(groundtruth_shape) != 1
            or len(prediction_shape) != 1
            or groundtruth_shape != prediction_shape
        ):
            raise ValueError(
                "A shape mismatch exists within the segmentation."
            )

        self.shape = groundtruth_shape.pop()
        self.size = int(np.prod(np.array(self.shape)))

        # Not sure if this is a requirement, really hard for a user to guarantee
        # combined_mask = np.concatenate(
        #     [prediction.mask for prediction in self.predictions],
        #     axis=0,
        # )
        # if not np.isclose(combined_mask.sum(axis=0), 1.0).all():
        #     raise ValueError("Segmentation scores must sum to 1.0.")

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Bitmask:
    mask: NDArray[np.bool_]
    label: str

    def __post_init__(self):
        if self.mask.dtype != np.bool_:
            raise ValueError(
                f"Bitmask recieved mask with dtype `{self.mask.dtype}`."
            )


@dataclass
class Segmentation:
    uid: str
    groundtruths: list[Bitmask]
    predictions: list[Bitmask]
    shape: tuple[int, ...] = field(default_factory=lambda: (0, 0))
    size: int = field(default=0)

    def __post_init__(self):

        groundtruth_shape = {
            groundtruth.mask.shape for groundtruth in self.groundtruths
        }
        prediction_shape = {
            prediction.mask.shape for prediction in self.predictions
        }
        if len(groundtruth_shape) == 0:
            raise ValueError("The segmenation is missing ground truths.")
        elif len(prediction_shape) == 0:
            raise ValueError("The segmenation is missing predictions.")
        elif (
            len(groundtruth_shape) != 1
            or len(prediction_shape) != 1
            or groundtruth_shape != prediction_shape
        ):
            raise ValueError(
                "A shape mismatch exists within the segmentation."
            )

        self.shape = groundtruth_shape.pop()
        self.size = int(np.prod(np.array(self.shape)))

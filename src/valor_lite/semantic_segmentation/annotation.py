from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Segmentation:
    """Ground truth and prediction label maps for one image.

    Parameters
    ----------
    uid : str
        Unique identifier for the image or sample.
    groundtruths : numpy.ndarray
        Nonempty 2D integer array. Zero is background; positive value i
        identifies ``labels[i - 1]``.
    predictions : numpy.ndarray
        Integer array with the same shape and label mapping as groundtruths.
    labels : list[str]
        Unique foreground class names, in local index order, with at most
        65,535 entries. For N pixel indices (0 through N-1), supply N-1 names.
        Background has no entry. An empty list allows only background pixels.
    metadata : dict[str, Any], optional
        Image metadata used for filtering.

    Notes
    -----
    Arrays are validated before conversion to uint16. Arrays already having
    that dtype may share storage with the inputs; validation and loading never
    modify their pixels. Background participates in accuracy but has no
    per-class metrics and is excluded from mean IoU.

    Examples
    --------
    >>> segmentation = Segmentation(
    ...     uid="image-1",
    ...     groundtruths=np.array([[0, 1], [1, 0]], dtype=np.uint16),
    ...     predictions=np.array([[0, 1], [0, 0]], dtype=np.uint16),
    ...     labels=["sky", "road"],
    ... )
    """

    uid: str
    groundtruths: NDArray[np.integer[Any]]
    predictions: NDArray[np.integer[Any]]
    labels: list[str]
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if not isinstance(self.labels, list) or not all(
            isinstance(label, str) for label in self.labels
        ):
            raise ValueError("labels must be a list of strings")
        if len(self.labels) > 65_535:
            raise ValueError("labels must contain at most 65,535 entries")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")

        self.groundtruths = self._validate_map(
            self.groundtruths, "groundtruths"
        )
        self.predictions = self._validate_map(self.predictions, "predictions")
        if self.groundtruths.shape != self.predictions.shape:
            raise ValueError(
                "groundtruths and predictions must have the same shape"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.groundtruths.shape[0], self.groundtruths.shape[1]

    @property
    def size(self) -> int:
        return self.groundtruths.size

    def _validate_map(
        self, array: NDArray[np.integer[Any]], name: str
    ) -> NDArray[np.uint16]:
        if not isinstance(array, np.ndarray) or not np.issubdtype(
            array.dtype, np.integer
        ):
            raise ValueError(f"{name} must be an integer NumPy array")
        if array.ndim != 2 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty 2D array")
        if array.min() < 0 or array.max() > len(self.labels):
            raise ValueError(
                f"{name} values must be between 0 (background) and len(labels)"
            )
        return array.astype(np.uint16, copy=False)

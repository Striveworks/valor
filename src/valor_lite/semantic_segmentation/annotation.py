from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Bitmask:
    """
    Represents a binary mask with an associated semantic label.

    Parameters
    ----------
    mask : NDArray[np.bool_]
        A NumPy array of boolean values representing the mask.
    label : str
        The semantic label associated with the mask.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    >>> bitmask = Bitmask(mask=mask, label='ocean')
    """

    mask: NDArray[np.bool_]
    label: str

    def __post_init__(self):
        if self.mask.dtype != np.bool_:
            raise ValueError(
                f"Bitmask recieved mask with dtype '{self.mask.dtype}'."
            )


@dataclass
class Segmentation:
    """
    Segmentation data structure holding ground truth and prediction bitmasks for semantic segmentation tasks.

    Parameters
    ----------
    uid : str
        Unique identifier for the image or sample.
    groundtruths : List[Bitmask]
        List of ground truth bitmasks.
    predictions : List[Bitmask]
        List of predicted bitmasks.
    shape : tuple of int, optional
        The shape of the segmentation masks. This is set automatically after initialization.
    size : int, optional
        The total number of pixels in the masks. This is set automatically after initialization.

    Examples
    --------
    >>> import numpy as np
    >>> mask1 = np.array([[True, False], [False, True]], dtype=np.bool_)
    >>> groundtruth = Bitmask(mask=mask1, label='object')
    >>> mask2 = np.array([[False, True], [True, False]], dtype=np.bool_)
    >>> prediction = Bitmask(mask=mask2, label='object')
    >>> segmentation = Segmentation(
    ...     uid='123',
    ...     groundtruths=[groundtruth],
    ...     predictions=[prediction]
    ... )
    """

    uid: str
    groundtruths: list[Bitmask]
    predictions: list[Bitmask]
    shape: tuple[int, ...]
    size: int = field(default=0)

    def __post_init__(self):

        if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError(
                f"segmentations must be 2-dimensional and have non-zero dimensions. Recieved shape '{self.shape}'"
            )
        self.size = self.shape[0] * self.shape[1]

        mask_accumulation = None
        for groundtruth in self.groundtruths:
            if self.shape != groundtruth.mask.shape:
                raise ValueError(
                    f"ground truth masks for datum '{self.uid}' should have shape '{self.shape}'. Received mask with shape '{groundtruth.mask.shape}'"
                )

            if mask_accumulation is None:
                mask_accumulation = groundtruth.mask.copy()
            elif np.logical_and(mask_accumulation, groundtruth.mask).any():
                raise ValueError("ground truth masks cannot overlap")
            else:
                mask_accumulation = mask_accumulation | groundtruth.mask

        mask_accumulation = None
        for prediction in self.predictions:
            if self.shape != prediction.mask.shape:
                raise ValueError(
                    f"prediction masks for datum '{self.uid}' should have shape '{self.shape}'. Received mask with shape '{prediction.mask.shape}'"
                )

            if mask_accumulation is None:
                mask_accumulation = prediction.mask.copy()
            elif np.logical_and(mask_accumulation, prediction.mask).any():
                raise ValueError("prediction masks cannot overlap")
            else:
                mask_accumulation = mask_accumulation | prediction.mask

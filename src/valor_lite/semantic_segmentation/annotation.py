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


def generate_segmentation(
    datum_uid: str,
    number_of_unique_labels: int,
    mask_height: int,
    mask_width: int,
) -> Segmentation:
    """
    Generates a semantic segmentation annotation.

    Parameters
    ----------
    datum_uid : str
        The datum UID for the generated segmentation.
    number_of_unique_labels : int
        The number of unique labels.
    mask_height : int
        The height of the mask in pixels.
    mask_width : int
        The width of the mask in pixels.

    Returns
    -------
    Segmentation
        A generated semantic segmenatation annotation.
    """

    if number_of_unique_labels > 1:
        common_proba = 0.4 / (number_of_unique_labels - 1)
        min_proba = min(common_proba, 0.1)
        labels = [str(i) for i in range(number_of_unique_labels)] + [None]
        proba = (
            [0.5]
            + [common_proba for _ in range(number_of_unique_labels - 1)]
            + [0.1]
        )
    elif number_of_unique_labels == 1:
        labels = ["0", None]
        proba = [0.9, 0.1]
        min_proba = 0.1
    else:
        raise ValueError(
            "The number of unique labels should be greater than zero."
        )

    probabilities = np.array(proba, dtype=np.float64)
    weights = (probabilities / min_proba).astype(np.int32)

    indices = np.random.choice(
        np.arange(len(weights)),
        size=(mask_height * 2, mask_width),
        p=probabilities,
    )

    N = len(labels)

    masks = np.arange(N)[:, None, None] == indices

    gts = []
    pds = []
    for lidx in range(N):
        label = labels[lidx]
        if label is None:
            continue
        gts.append(
            Bitmask(
                mask=masks[lidx, :mask_height, :],
                label=label,
            )
        )
        pds.append(
            Bitmask(
                mask=masks[lidx, mask_height:, :],
                label=label,
            )
        )

    return Segmentation(
        uid=datum_uid,
        groundtruths=gts,
        predictions=pds,
    )

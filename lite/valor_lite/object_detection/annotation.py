from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class BoundingBox:
    """
    Represents a bounding box with associated labels and optional scores.

    Parameters
    ----------
    xmin : float
        The minimum x-coordinate of the bounding box.
    xmax : float
        The maximum x-coordinate of the bounding box.
    ymin : float
        The minimum y-coordinate of the bounding box.
    ymax : float
        The maximum y-coordinate of the bounding box.
    labels : list of str
        List of labels associated with the bounding box.
    scores : list of float, optional
        Confidence scores corresponding to each label. Defaults to an empty list.

    Examples
    --------
    Ground Truth Example:

    >>> bbox = BoundingBox(xmin=10.0, xmax=50.0, ymin=20.0, ymax=60.0, labels=['cat'])

    Prediction Example:

    >>> bbox = BoundingBox(
    ...     xmin=10.0, xmax=50.0, ymin=20.0, ymax=60.0,
    ...     labels=['cat', 'dog'], scores=[0.9, 0.1]
    ... )
    """

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
        Returns the bounding box extrema.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple in the form (xmin, xmax, ymin, ymax).
        """
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def annotation(self) -> tuple[float, float, float, float]:
        """
        Returns the annotation's data representation.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple in the form (xmin, xmax, ymin, ymax).
        """
        return self.extrema


@dataclass
class Polygon:
    """
    Represents a polygon shape with associated labels and optional scores.

    Parameters
    ----------
    shape : ShapelyPolygon
        A Shapely polygon object representing the shape.
    labels : list of str
        List of labels associated with the polygon.
    scores : list of float, optional
        Confidence scores corresponding to each label. Defaults to an empty list.

    Examples
    --------
    Ground Truth Example:

    >>> from shapely.geometry import Polygon as ShapelyPolygon
    >>> shape = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> polygon = Polygon(shape=shape, labels=['building'])

    Prediction Example:

    >>> polygon = Polygon(
    ...     shape=shape, labels=['building'], scores=[0.95]
    ... )
    """

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
        Returns the polygon's bounding box extrema.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple in the form (xmin, xmax, ymin, ymax).
        """
        xmin, ymin, xmax, ymax = self.shape.bounds
        return (xmin, xmax, ymin, ymax)

    @property
    def annotation(self) -> ShapelyPolygon:
        """
        Returns the annotation's data representation.

        Returns
        -------
        shapely.geometry.Polygon
            The polygon shape.
        """
        return self.shape


@dataclass
class Bitmask:
    """
    Represents a binary mask with associated labels and optional scores.

    Parameters
    ----------
    mask : NDArray[np.bool_]
        A NumPy array of boolean values representing the mask.
    labels : list of str
        List of labels associated with the mask.
    scores : list of float, optional
        Confidence scores corresponding to each label. Defaults to an empty list.

    Examples
    --------
    Ground Truth Example:

    >>> import numpy as np
    >>> mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    >>> bitmask = Bitmask(mask=mask, labels=['tree'])

    Prediction Example:

    >>> bitmask = Bitmask(
    ...     mask=mask, labels=['tree'], scores=[0.85]
    ... )
    """

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
        Returns the bounding box extrema of the mask.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple in the form (xmin, xmax, ymin, ymax).
        """
        rows, cols = np.nonzero(self.mask)
        return (cols.min(), cols.max(), rows.min(), rows.max())

    @property
    def annotation(self) -> NDArray[np.bool_]:
        """
        Returns the annotation's data representation.

        Returns
        -------
        NDArray[np.bool_]
            The binary mask array.
        """
        return self.mask


@dataclass
class Detection:
    """
    Detection data structure holding ground truths and predictions for object detection tasks.

    Parameters
    ----------
    uid : str
        Unique identifier for the image or sample.
    groundtruths : list of BoundingBox or Bitmask or Polygon
        List of ground truth annotations.
    predictions : list of BoundingBox or Bitmask or Polygon
        List of predicted annotations.

    Examples
    --------
    >>> bbox_gt = BoundingBox(xmin=10, xmax=50, ymin=20, ymax=60, labels=['cat'])
    >>> bbox_pred = BoundingBox(
    ...     xmin=12, xmax=48, ymin=22, ymax=58, labels=['cat'], scores=[0.9]
    ... )
    >>> detection = Detection(
    ...     uid='image_001',
    ...     groundtruths=[bbox_gt],
    ...     predictions=[bbox_pred]
    ... )
    """

    uid: str
    groundtruths: list[BoundingBox] | list[Bitmask] | list[Polygon]
    predictions: list[BoundingBox] | list[Bitmask] | list[Polygon]

    def __post_init__(self):
        for prediction in self.predictions:
            if len(prediction.scores) != len(prediction.labels):
                raise ValueError(
                    "Predictions must provide a score for every label."
                )

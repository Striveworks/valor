from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class BoundingBox:
    """
    Represents a bounding box with associated labels and optional scores.

    Parameters
    ----------
    uid : str
        A unique identifier.
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

    >>> bbox = BoundingBox(uid="xyz", xmin=10.0, xmax=50.0, ymin=20.0, ymax=60.0, labels=['cat'])

    Prediction Example:

    >>> bbox = BoundingBox(
    ...     uid="abc",
    ...     xmin=10.0, xmax=50.0, ymin=20.0, ymax=60.0,
    ...     labels=['cat', 'dog'], scores=[0.9, 0.1]
    ... )
    """

    uid: str
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
        Returns the annotation's data representation.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple in the form (xmin, xmax, ymin, ymax).
        """
        return (self.xmin, self.xmax, self.ymin, self.ymax)


@dataclass
class Polygon:
    """
    Represents a polygon shape with associated labels and optional scores.

    Parameters
    ----------
    uid : str
        A unique identifier.
    shape : shapely.geometry.Polygon
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
    >>> polygon = Polygon(uid="xyz", shape=shape, labels=['building'])

    Prediction Example:

    >>> polygon = Polygon(
    ...     uid="abc", shape=shape, labels=['building'], scores=[0.95]
    ... )
    """

    uid: str
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


@dataclass
class Bitmask:
    """
    Represents a binary mask with associated labels and optional scores.

    Parameters
    ----------
    uid : str
        A unique identifier.
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
    >>> bitmask = Bitmask(uid="abc", mask=mask, labels=['tree'])

    Prediction Example:

    >>> bitmask = Bitmask(
    ...     uid="xyz", mask=mask, labels=['tree'], scores=[0.85]
    ... )
    """

    uid: str
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


AnnotationType = TypeVar("AnnotationType", BoundingBox, Polygon, Bitmask)


@dataclass
class Detection(Generic[AnnotationType]):
    """
    Detection data structure holding ground truths and predictions for object detection tasks.

    Parameters
    ----------
    uid : str
        Unique identifier for the image or sample.
    groundtruths : list[BoundingBox] | list[Polygon] | list[Bitmask]
        List of ground truth annotations.
    predictions : list[BoundingBox] | list[Polygon] | list[Bitmask]
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
    groundtruths: list[AnnotationType]
    predictions: list[AnnotationType]

    def __post_init__(self):
        for prediction in self.predictions:
            if len(prediction.scores) != len(prediction.labels):
                raise ValueError(
                    "Predictions must provide a score for every label."
                )

from typing import List, Tuple, Union

import numpy as np
import shapely.affinity
from shapely.geometry import Polygon as ShapelyPolygon

# turn off "invalid value encountered in scalar divide" warning
# when dividing by 0 or NaN, the returned value will be NaN. we'll then handle those NaNs later in the evaluation code
np.seterr(divide="ignore", invalid="ignore")


def calculate_bbox_iou(
    bbox1: List[Tuple[float, float]], bbox2: List[Tuple[float, float]]
) -> float:
    """
    Calculate the Intersection over Union (IOU) for two bounding boxes.

    Parameters
    ----------
    bbox1 : List[Tuple[float, float]]
        Coordinates of the first bounding box.
    bbox2 : List[Tuple[float, float]]
        Coordinates of the second bounding box.

    Returns
    ----------
    float
        The IOU value between 0 and 1.
    """
    poly1 = ShapelyPolygon(bbox1)
    poly2 = ShapelyPolygon(bbox2)
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    return intersection_area / union_area if union_area != 0 else 0


def rotate_bbox(
    bbox: List[Tuple[float, float]],
    angle: float,
    origin: Union[str, Tuple[float, float]] = "centroid",
) -> List[Tuple[float, float]]:
    """
    Rotate a bounding box by a given angle around the centroid of a polygon.

    Parameters
    ----------
    bbox : List[Tuple[float, float]]
        Coordinates of the bounding box.
    angle : float
        The rotation angle in degrees.
    origin : Union[str, Tuple[float, float]]
        The point around which to rotate. Default is "centroid".

    Returns
    ----------
    List[Tuple[float, float]]
        Coordinates of the rotated bounding box.
    """
    return list(
        shapely.affinity.rotate(
            ShapelyPolygon(bbox), angle=angle, origin=origin  # type: ignore - shapely type error. can be a string ("centroid", "center") or a tuple of coordinates
        ).exterior.coords
    )


def is_axis_aligned(bbox: List[Tuple[float, float]]) -> bool:
    """
    Check if the bounding box is axis-aligned.

    Parameters
    ----------
    bbox : List[Tuple[float, float]]
        Coordinates of the bounding box.

    Returns
    ----------
    bool
        True if the bounding box is axis-aligned, otherwise False.
    """
    return all(
        x1 == x2 or y1 == y2
        for (x1, y1), (x2, y2) in zip(bbox, bbox[1:] + bbox[:1])
    )


def is_skewed(bbox: List[Tuple[float, float]]) -> bool:
    """
    Check if the bounding box is skewed.

    Parameters
    ----------
    bbox : List[Tuple[float, float]]
        Coordinates of the bounding box.

    Returns
    ----------
    bool
        True if the bounding box is skewed, otherwise False.
    """

    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = dot_product / norm_product
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    vectors = []
    for (x1, y1), (x2, y2) in zip(bbox, bbox[1:] + bbox[:1]):
        vectors.append(np.array([x2 - x1, y2 - y1]))

    angles = [
        angle_between(vectors[i], vectors[(i + 1) % len(vectors)])
        for i in range(len(vectors))
    ]

    return not all(
        np.isclose(angle, np.pi / 2, atol=1e-2)  # if close to 90 degrees
        for angle in angles
        if not np.isnan(angle)
    )


def is_rotated(bbox: List[Tuple[float, float]]) -> bool:
    """
    Check if the bounding box is rotated (not axis-aligned and not skewed).

    Parameters
    ----------
    bbox : List[Tuple[float, float]]
        Coordinates of the bounding box.

    Returns
    ----------
    bool
        True if the bounding box is rotated, otherwise False.
    """
    return not is_axis_aligned(bbox) and not is_skewed(bbox)

import numpy as np
import pandas as pd
import shapely.affinity
from shapely.geometry import Polygon as ShapelyPolygon

# turn off "invalid value encountered in scalar divide" warning
# when dividing by 0 or NaN, the returned value will be NaN. we'll then handle those NaNs later in the evaluation code
np.seterr(divide="ignore", invalid="ignore")


def _calculate_area_and_boundaries_of_bbox_series(array):
    """Calculate the area and boundaries for each bbox represented in a numpy array."""
    xmin = np.min(array[:, :, 0], axis=1)
    xmax = np.max(array[:, :, 0], axis=1)
    ymin = np.min(array[:, :, 1], axis=1)
    ymax = np.max(array[:, :, 1], axis=1)

    area = (ymax - ymin) * (xmax - xmin)

    return area, (xmin, xmax, ymin, ymax)


def calculate_axis_aligned_bbox_iou(
    series_of_bboxes1: pd.Series, series_of_bboxes2: pd.Series
) -> pd.Series:
    """
    Calculate the IOU between two series of axis-aligned bounding boxes.

    Parameters
    ----------
    series_of_bboxes1 : pd.Series
        Series of bounding boxes, where each element is an array-like object representing a bounding box.
    series_of_bboxes2 : pd.Series
        Series of bounding boxes with the same format as series1.

    Returns
    -------
    pd.Series
        Series containing the IOU for each pair of bounding boxes.
    """
    # convert series to NumPy arrays for vectorized operations. note that this output has a different shape than using .to_numpy()
    series1 = np.array(series_of_bboxes1.tolist())
    series2 = np.array(series_of_bboxes2.tolist())

    s1_area, (
        s1_xmin,
        s1_xmax,
        s1_ymin,
        s1_ymax,
    ) = _calculate_area_and_boundaries_of_bbox_series(series1)
    s2_area, (
        s2_xmin,
        s2_xmax,
        s2_ymin,
        s2_ymax,
    ) = _calculate_area_and_boundaries_of_bbox_series(series2)

    intersection_width = np.clip(
        np.minimum(s1_xmax, s2_xmax) - np.maximum(s1_xmin, s2_xmin), 0, None
    )
    intersection_height = np.clip(
        np.minimum(s1_ymax, s2_ymax) - np.maximum(s1_ymin, s2_ymin), 0, None
    )

    intersection = intersection_height * intersection_width
    union = s1_area + s2_area - intersection

    iou = intersection / union

    # the indexes of series_of_bboxes1 and series_of_bboxes2 are the same, so it doesn't matter which you use
    return pd.Series(iou, index=series_of_bboxes1.index)


def calculate_iou(
    bbox1: list[tuple[float, float]], bbox2: list[tuple[float, float]]
) -> float:
    """
    Calculate the Intersection over Union (IOU) for two bounding boxes.

    Parameters
    ----------
    bbox1 : list[tuple[float, float]]
        Coordinates of the first bounding box.
    bbox2 : list[tuple[float, float]]
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
    bbox: list[tuple[float, float]],
    angle: float,
    origin: str | tuple[float, float] = "centroid",
) -> list[tuple[float, float]]:
    """
    Rotate a bounding box by a given angle around the centroid of a polygon.

    Parameters
    ----------
    bbox : list[tuple[float, float]]
        Coordinates of the bounding box.
    angle : float
        The rotation angle in degrees.
    origin : str | tuple[float, float]
        The point around which to rotate. Default is "centroid".

    Returns
    ----------
    list[tuple[float, float]]
        Coordinates of the rotated bounding box.
    """
    return list(
        shapely.affinity.rotate(
            ShapelyPolygon(bbox), angle=angle, origin=origin  # type: ignore - shapely type error. can be a string ("centroid", "center") or a tuple of coordinates
        ).exterior.coords
    )


def is_axis_aligned(bbox: list[tuple[float, float]]) -> bool:
    """
    Check if the bounding box is axis-aligned.

    Parameters
    ----------
    bbox : list[tuple[float, float]]
        Coordinates of the bounding box.

    Returns
    ----------
    bool
        True if the bounding box is axis-aligned, otherwise False.
    """

    if isinstance(bbox, np.ndarray):
        raise ValueError(
            "Please make sure your bounding box is a list, otherwise is_axis_aligned may not work correctly."
        )

    return all(
        x1 == x2 or y1 == y2
        for (x1, y1), (x2, y2) in zip(bbox, bbox[1:] + bbox[:1])
    )


def is_skewed(bbox: list[tuple[float, float]]) -> bool:
    """
    Check if the bounding box is skewed.

    Parameters
    ----------
    bbox : list[tuple[float, float]]
        Coordinates of the bounding box.

    Returns
    ----------
    bool
        True if the bounding box is skewed, otherwise False.
    """

    def _calculate_angle_between_arrays(
        v1: np.ndarray, v2: np.ndarray
    ) -> float:
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = dot_product / norm_product
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    vectors = []
    for (x1, y1), (x2, y2) in zip(bbox, bbox[1:] + bbox[:1]):
        vectors.append(np.array([x2 - x1, y2 - y1]))

    angles = [
        _calculate_angle_between_arrays(
            vectors[i], vectors[(i + 1) % len(vectors)]
        )
        for i in range(len(vectors))
    ]

    return not all(
        np.isclose(angle, np.pi / 2, atol=1e-2)  # if close to 90 degrees
        for angle in angles
        if not np.isnan(angle)
    )


def is_rotated(bbox: list[tuple[float, float]]) -> bool:
    """
    Check if the bounding box is rotated (not axis-aligned and not skewed).

    Parameters
    ----------
    bbox : list[tuple[float, float]]
        Coordinates of the bounding box.

    Returns
    ----------
    bool
        True if the bounding box is rotated, otherwise False.
    """
    return not is_axis_aligned(bbox) and not is_skewed(bbox)


def calculate_raster_ious(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Calculate the IOUs between two series of rasters.

    Parameters
    ----------
    series1 : pd.Series
        The first series of rasters.
    series2: pd.Series
        The second series of rasters.

    Returns
    ----------
    pd.Series
        A Series of IOUs.
    """

    if len(series1) != len(series2):
        raise ValueError(
            "Series of rasters must be the same length to calculate IOUs."
        )

    intersection_ = pd.Series(
        [np.logical_and(x, y).sum() for x, y in zip(series1, series2)]
    )

    union_ = pd.Series(
        [np.logical_or(x, y).sum() for x, y in zip(series1, series2)]
    )

    if (intersection_ > union_).any():
        raise ValueError("Intersection can't be greater than union.")

    return intersection_ / union_

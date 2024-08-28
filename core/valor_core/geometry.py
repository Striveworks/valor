import numba
import numpy as np
import pandas as pd
import shapely.affinity
from shapely.geometry import Polygon as ShapelyPolygon

# turn off "invalid value encountered in scalar divide" warning
# when dividing by 0 or NaN, the returned value will be NaN. we'll then handle those NaNs later in the evaluation code
np.seterr(divide="ignore", invalid="ignore")


@numba.jit(nopython=True)
def calculate_axis_aligned_bbox_intersection(
    bbox1: np.ndarray, bbox2: np.ndarray
) -> float:
    """
    Calculate the intersection area between two axis-aligned bounding boxes.

    Parameters
    ----------
    bbox1 : np.ndarray
        Array representing the first bounding bo.
    bbox2 : np.ndarray
        Array representing the second bounding box with the same format as bbox1.
    Returns
    -------
    float
        The area of the intersection between the two bounding boxes.
    Raises
    ------
    ValueError
        If the input bounding boxes do not have the expected shape.
    """

    xmin_inter = max(bbox1[:, 0].min(), bbox2[:, 0].min())
    ymin_inter = max(bbox1[:, 1].min(), bbox2[:, 1].min())
    xmax_inter = min(bbox1[:, 0].max(), bbox2[:, 0].max())
    ymax_inter = min(bbox1[:, 1].max(), bbox2[:, 1].max())

    width = max(0, xmax_inter - xmin_inter)
    height = max(0, ymax_inter - ymin_inter)

    intersection_area = width * height

    return intersection_area


@numba.jit(nopython=True)
def calculate_axis_aligned_bbox_union(
    bbox1: np.ndarray, bbox2: np.ndarray
) -> float:
    """
    Calculate the union area between two axis-aligned bounding boxes.

    Parameters
    ----------
    bbox1 : np.ndarray
        Array representing the first bounding box.
    bbox2 : np.ndarray
        Array representing the second bounding box with the same format as bbox1.
    Returns
    -------
    float
        The area of the union between the two bounding boxes.
    Raises
    ------
    ValueError
        If the input bounding boxes do not have the expected shape.
    """
    area1 = (bbox1[:, 0].max() - bbox1[:, 0].min()) * (
        bbox1[:, 1].max() - bbox1[:, 1].min()
    )
    area2 = (bbox2[:, 0].max() - bbox2[:, 0].min()) * (
        bbox2[:, 1].max() - bbox2[:, 1].min()
    )
    union_area = (
        area1 + area2 - calculate_axis_aligned_bbox_intersection(bbox1, bbox2)
    )
    return union_area


@numba.jit(nopython=True)
def calculate_axis_aligned_bbox_iou(
    bbox1: np.ndarray, bbox2: np.ndarray
) -> float:
    """
    Calculate the Intersection over Union (IoU) between two axis-aligned bounding boxes.

    Parameters
    ----------
    bbox1 : np.ndarray
        Array representing the first bounding box.
    bbox2 : np.ndarray
        Array representing the second bounding box with the same format as bbox1.
    Returns
    -------
    float
        The IoU between the two bounding boxes. Returns 0 if the union area is zero.

    Raises
    ------
    ValueError
        If the input bounding boxes do not have the expected shape.
    """
    intersection = calculate_axis_aligned_bbox_intersection(bbox1, bbox2)
    union = calculate_axis_aligned_bbox_union(bbox1, bbox2)
    iou = intersection / union

    return iou


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


def calculate_raster_intersection(row: pd.Series) -> pd.Series:
    """
    Calculate the raster intersection for a given row in a pandas DataFrame. This function is intended to be used with .apply.

    Parameters
    ----------
    row : pd.Series
        A row of a pandas.DataFrame containing two masks in the columns "converted_geometry_pd" and "converted_geometry_gt".

    Returns
    ----------
    pd.Series
        A Series indicating the intersection of two masks.
    """
    return np.logical_and(
        row["converted_geometry_pd"], row["converted_geometry_gt"]
    ).sum()


def calculate_raster_union(row: pd.Series) -> pd.Series:
    """
    Calculate the raster union for a given row in a pandas DataFrame. This function is intended to be used with .apply.

    Parameters
    ----------
    row : pd.Series
        A row of a pandas.DataFrame containing two masks in the columns "converted_geometry_pd" and "converted_geometry_gt".

    Returns
    ----------
    pd.Series
        A Series indicating the union of two masks.
    """

    return np.sum(row["converted_geometry_gt"]) + np.sum(
        row["converted_geometry_pd"]
    )

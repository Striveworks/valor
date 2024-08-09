import numpy as np


def calculate_bbox_intersection(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the intersection area between two bounding boxes.

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
    # Calculate intersection coordinates
    xmin_inter = max(bbox1[:, 0].min(), bbox2[:, 0].min())
    ymin_inter = max(bbox1[:, 1].min(), bbox2[:, 1].min())
    xmax_inter = min(bbox1[:, 0].max(), bbox2[:, 0].max())
    ymax_inter = min(bbox1[:, 1].max(), bbox2[:, 1].max())

    # Calculate width and height of intersection area
    width = max(0, xmax_inter - xmin_inter)
    height = max(0, ymax_inter - ymin_inter)

    # Calculate intersection area
    intersection_area = width * height
    return intersection_area


def calculate_bbox_union(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the union area between two bounding boxes.

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
    union_area = area1 + area2 - calculate_bbox_intersection(bbox1, bbox2)
    return union_area


def calculate_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

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
    intersection = calculate_bbox_intersection(bbox1, bbox2)
    union = calculate_bbox_union(bbox1, bbox2)
    iou = intersection / union
    return iou

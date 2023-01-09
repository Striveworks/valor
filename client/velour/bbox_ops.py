from typing import List

from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    PredictedDetection,
)


def area(rect: BoundingPolygon) -> float:
    assert len(rect.points) == 4
    xs = [pt.x for pt in rect.points]
    ys = [pt.y for pt in rect.points]

    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    assert len(rect1.points) == len(rect2.points) == 4

    xs1 = [pt.x for pt in rect1.points]
    xs2 = [pt.x for pt in rect2.points]

    ys1 = [pt.y for pt in rect1.points]
    ys2 = [pt.y for pt in rect2.points]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


def _match_array(
    ious: List[List[float]],
    iou_thres: float,
) -> List[int | None]:
    """
    iou[i][j] is iou between predicted detection i and groundtruth detection j

    important assumption is that the predicted detections are sorted by confidence

    Returns
    -------
    array of match indices. value at index i is the index of the groundtruth object that
    best matches the ith detection, or None if that deteciton has no match
    """
    matches = []
    for i in range(len(ious)):
        best_match_idx, current_max_iou = None, None
        for j, iou in enumerate(ious[i]):
            if iou >= iou_thres and j not in matches:
                if best_match_idx is None or iou > current_max_iou:
                    best_match_idx = j
                    current_max_iou = iou

        matches.append(best_match_idx)

    return matches


def iou_matrix(
    groundtruths: List[GroundTruthDetection],
    predictions: List[PredictedDetection],
) -> List[List[float]]:
    return [
        [iou(p.boundary, g.boundary) for g in groundtruths]
        for p in predictions
    ]

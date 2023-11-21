from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend.core import get_dataset, get_model
from velour_api.backend.metrics.detection import compute_detection_metrics
from velour_api.backend.models import GroundTruth, Prediction


def _round_dict(d: dict, prec: int = 3) -> None:
    """Modifies a dictionary in place by rounding every float in it
    to three decimal places
    """
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, prec)
        elif isinstance(v, dict):
            _round_dict(v, prec)


def test_compute_detection_metrics(
    db: Session,
    groundtruths: list[list[GroundTruth]],
    predictions: list[list[Prediction]],
):
    iou_thresholds = set([round(0.5 + 0.05 * i, 2) for i in range(10)])
    metrics = compute_detection_metrics(
        db=db,
        dataset=get_dataset(db, "test_dataset"),
        model=get_model(db, "test_model"),
        target_type=enums.AnnotationType.BOX,
        settings=schemas.EvaluationSettings(
            parameters=schemas.DetectionParameters(
                iou_thresholds_to_compute=iou_thresholds,
                iou_thresholds_to_keep=[0.5, 0.75],
            ),
            filters=schemas.Filter(
                annotation_types=[enums.AnnotationType.BOX],
                label_keys=["class"],
            ),
        ),
    )

    metrics = [m.model_dump(exclude_none=True) for m in metrics]

    for m in metrics:
        _round_dict(m, 3)

    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    expected = [
        # AP METRICS
        {"iou": 0.5, "value": 0.505, "label": {"key": "class", "value": "2"}},
        {"iou": 0.75, "value": 0.505, "label": {"key": "class", "value": "2"}},
        {"iou": 0.5, "value": 0.79, "label": {"key": "class", "value": "49"}},
        {
            "iou": 0.75,
            "value": 0.576,
            "label": {"key": "class", "value": "49"},
        },
        # {"iou": 0.5, "value": -1.0, "label": {"key": "class", "value": "3"}},
        # {"iou": 0.75, "value": -1.0, "label": {"key": "class", "value": "3"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "0"}},
        {"iou": 0.75, "value": 0.723, "label": {"key": "class", "value": "0"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "1"}},
        {"iou": 0.75, "value": 1.0, "label": {"key": "class", "value": "1"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "4"}},
        {"iou": 0.75, "value": 1.0, "label": {"key": "class", "value": "4"}},
        # mAP METRICS
        {"iou": 0.5, "value": 0.859},
        {"iou": 0.75, "value": 0.761},
        # AP METRICS AVERAGED OVER IOUS
        {
            "ious": iou_thresholds,
            "value": 0.454,
            "label": {"key": "class", "value": "2"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.555,  # note COCO had 0.556
            "label": {"key": "class", "value": "49"},
        },
        # {
        #     "ious": iou_thresholds,
        #     "value": -1.0,
        #     "label": {"key": "class", "value": "3"},
        # },
        {
            "ious": iou_thresholds,
            "value": 0.725,
            "label": {"key": "class", "value": "0"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.8,
            "label": {"key": "class", "value": "1"},
        },
        {
            "ious": iou_thresholds,
            "value": 0.650,
            "label": {"key": "class", "value": "4"},
        },
        # mAP METRICS AVERAGED OVER IOUS
        {"ious": iou_thresholds, "value": 0.637},
    ]

    assert len(metrics) == len(expected)

    # sort labels lists
    for m in metrics + expected:
        if "labels" in m:
            m["labels"] = sorted(m["labels"], key=lambda x: x["value"])

    # check that metrics and labels are equivalent
    for m in metrics:
        assert m in expected

    for m in expected:
        assert m in metrics

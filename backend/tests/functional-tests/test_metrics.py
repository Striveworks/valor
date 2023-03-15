from sqlalchemy.orm import Session

from velour_api.metrics import compute_ap_metrics, compute_map_metrics_from_aps
from velour_api.schemas import GroundTruthDetection, PredictedDetection


def round_dict_(d: dict, prec: int) -> None:
    """Modifies a dictionary in place by rounding every float in it
    to three decimal places
    """
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, prec)
        elif isinstance(v, dict):
            round_dict_(v, prec)


def test_compute_ap_metrics(
    db: Session,
    groundtruths: list[list[GroundTruthDetection]],
    predictions: list[list[PredictedDetection]],
):
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap_metrics = compute_ap_metrics(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        iou_thresholds=iou_thresholds,
    )
    map_metrics = compute_map_metrics_from_aps(ap_metrics)
    metrics = ap_metrics + map_metrics

    # only look at APs at thresholds 0.5 and 0.75 or averaged over all iou thresholds
    metrics = [
        m for m in metrics if m.iou in [0.5, 0.75] or isinstance(m.iou, list)
    ]

    # convert to dicts and round
    metrics = [m.dict() for m in metrics]
    for m in metrics:
        round_dict_(m, 3)

    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    expected = [
        {"iou": 0.5, "value": 0.505, "label": {"key": "class", "value": "2"}},
        {"iou": 0.75, "value": 0.505, "label": {"key": "class", "value": "2"}},
        {"iou": 0.5, "value": 0.79, "label": {"key": "class", "value": "49"}},
        {
            "iou": 0.75,
            "value": 0.576,
            "label": {"key": "class", "value": "49"},
        },
        {"iou": 0.5, "value": -1.0, "label": {"key": "class", "value": "3"}},
        {"iou": 0.75, "value": -1.0, "label": {"key": "class", "value": "3"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "0"}},
        {"iou": 0.75, "value": 0.723, "label": {"key": "class", "value": "0"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "1"}},
        {"iou": 0.75, "value": 1.0, "label": {"key": "class", "value": "1"}},
        {"iou": 0.5, "value": 1.0, "label": {"key": "class", "value": "4"}},
        {"iou": 0.75, "value": 1.0, "label": {"key": "class", "value": "4"}},
        {
            "iou": 0.5,
            "value": 0.859,
            "labels": [
                {"key": "class", "value": "3"},
                {"key": "class", "value": "2"},
                {"key": "class", "value": "4"},
                {"key": "class", "value": "0"},
                {"key": "class", "value": "1"},
                {"key": "class", "value": "49"},
            ],
        },
        {
            "iou": 0.75,
            "value": 0.761,
            "labels": [
                {"key": "class", "value": "3"},
                {"key": "class", "value": "2"},
                {"key": "class", "value": "4"},
                {"key": "class", "value": "0"},
                {"key": "class", "value": "1"},
                {"key": "class", "value": "49"},
            ],
        },
    ]

    # sort labels lists
    for m in metrics + expected:
        if "labels" in m:
            m["labels"] = sorted(m["labels"], key=lambda x: x["value"])

    # check that metrics and labels are equivalent
    for m in metrics:
        assert m in expected

    for m in expected:
        assert m in metrics

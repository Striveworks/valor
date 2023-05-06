from sqlalchemy.orm import Session

from velour_api import crud, schemas
from velour_api.metrics import compute_ap_metrics
from velour_api.metrics.classification import get_hard_preds_from_label_key
from velour_api.models import (
    LabeledGroundTruthDetection,
    LabeledPredictedDetection,
)


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
    groundtruths: list[list[LabeledGroundTruthDetection]],
    predictions: list[list[LabeledPredictedDetection]],
):
    iou_thresholds = set([round(0.5 + 0.05 * i, 2) for i in range(10)])
    metrics = compute_ap_metrics(
        db=db,
        predictions=predictions,
        groundtruths=groundtruths,
        iou_thresholds=iou_thresholds,
        ious_to_keep=[0.5, 0.75],
    )

    metrics = [m.dict() for m in metrics]
    for m in metrics:
        round_dict_(m, 3)

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
        {"iou": 0.5, "value": -1.0, "label": {"key": "class", "value": "3"}},
        {"iou": 0.75, "value": -1.0, "label": {"key": "class", "value": "3"}},
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
        {
            "ious": iou_thresholds,
            "value": -1.0,
            "label": {"key": "class", "value": "3"},
        },
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

    # sort labels lists
    for m in metrics + expected:
        if "labels" in m:
            m["labels"] = sorted(m["labels"], key=lambda x: x["value"])

    # check that metrics and labels are equivalent
    for m in metrics:
        assert m in expected

    for m in expected:
        assert m in metrics


def test_get_hard_preds_from_label_key(db: Session):
    dataset_name = "test dataset"
    model_name = "test model"
    crud.create_dataset(db, schemas.DatasetCreate(name=dataset_name))
    crud.create_model(db, schemas.Model(name=model_name))

    animal_gts = ["bird", "dog", "bird", "bird", "cat", "dog"]
    animal_preds = [
        {"bird": 0.6, "dog": 0.2, "cat": 0.2},
        {"cat": 0.9, "dog": 0.1, "bird": 0.0},
        {"cat": 0.8, "dog": 0.05, "bird": 0.15},
        {"dog": 0.75, "cat": 0.1, "bird": 0.15},
        {"cat": 1.0, "dog": 0.0, "bird": 0.0},
        {"cat": 0.4, "dog": 0.4, "bird": 0.2},
    ]

    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_preds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

    imgs = [
        schemas.Image(uid=f"uid{i}", height=128, width=256) for i in range(6)
    ]

    gts = [
        schemas.GroundTruthImageClassification(
            image=imgs[i],
            labels=[
                schemas.Label(key="animal", value=animal_gts[i]),
                schemas.Label(key="color", value=color_gts[i]),
            ],
        )
        for i in range(6)
    ]
    preds = [
        schemas.PredictedImageClassification(
            image=imgs[i],
            scored_labels=[
                schemas.ScoredLabel(
                    label=schemas.Label(key="animal", value=value), score=score
                )
                for value, score in animal_preds[i].items()
            ]
            + [
                schemas.ScoredLabel(
                    label=schemas.Label(key="color", value=value), score=score
                )
                for value, score in color_preds[i].items()
            ],
        )
        for i in range(6)
    ]

    crud.create_ground_truth_image_classifications(
        db,
        data=schemas.GroundTruthImageClassificationsCreate(
            dataset_name=dataset_name, classifications=gts
        ),
    )
    crud.create_predicted_image_classifications(
        db,
        data=schemas.PredictedImageClassificationsCreate(
            model_name=model_name,
            dataset_name=dataset_name,
            classifications=preds,
        ),
    )

    label_key = "animal"
    preds = get_hard_preds_from_label_key(db, dataset_name, label_key)
    pred_values = [p.label.value for p in preds]
    assert pred_values[:5] == ["bird", "cat", "cat", "dog", "cat"]
    # last one could be dog or cat
    assert pred_values[-1] in ["cat", "dog"]

    label_key = "color"
    preds = get_hard_preds_from_label_key(db, dataset_name, label_key)
    pred_values = [p.label.value for p in preds]
    assert pred_values == ["white", "blue", "red", "white", "red", "red"]

    # maybe these two tests the `get_hard_preds_from_label_key` should probably
    # throw an error instead
    preds = get_hard_preds_from_label_key(db, "not a dataset", label_key)
    assert preds == []

    preds = get_hard_preds_from_label_key(db, dataset_name, "not a label key")
    assert preds == []

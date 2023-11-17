import pytest
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend.core import get_dataset, get_model
from velour_api.backend.metrics.classification import (
    accuracy_from_cm,
    confusion_matrix_at_label_key,
    roc_auc,
)
from velour_api.backend.metrics.detection import compute_detection_metrics
from velour_api.backend.metrics.segmentation import (
    _gt_query,
    _pred_query,
    compute_segmentation_metrics,
    get_groundtruth_labels,
    gt_count,
    pred_count,
    tp_count,
)
from velour_api.backend.models import GroundTruth, Label, Prediction

dataset_name = "test_dataset"
model_name = "test_model"


@pytest.fixture
def classification_test_data(db: Session):
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
        schemas.Datum(
            dataset=dataset_name,
            uid=f"uid{i}",
            metadata={
                "height": 128,
                "width": 256,
                "md1": f"md1-val{int(i == 4)}",
                "md2": f"md1-val{i % 3}",
            },
        )
        for i in range(6)
    ]

    gts = [
        schemas.GroundTruth(
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="animal", value=animal_gts[i]),
                        schemas.Label(key="color", value=color_gts[i]),
                    ],
                )
            ],
        )
        for i in range(6)
    ]

    preds = [
        schemas.Prediction(
            model=model_name,
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="animal", value=value, score=score)
                        for value, score in animal_preds[i].items()
                    ]
                    + [
                        schemas.Label(key="color", value=value, score=score)
                        for value, score in color_preds[i].items()
                    ],
                )
            ],
        )
        for i in range(6)
    ]

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dataset_name,
            metadata={"type": enums.DataType.IMAGE.value},
        ),
    )
    for gt in gts:
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name, metadata={"type": enums.DataType.IMAGE}
        ),
    )
    for pd in preds:
        crud.create_prediction(db=db, prediction=pd)


def round_dict_(d: dict, prec: int) -> None:
    """Modifies a dictionary in place by rounding every float in it
    to three decimal places
    """
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, prec)
        elif isinstance(v, dict):
            round_dict_(v, prec)


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
        gt_type=enums.AnnotationType.BOX,
        pd_type=enums.AnnotationType.BOX,
        settings=schemas.EvaluationSettings(
            task_type=enums.TaskType.DETECTION,
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


def test_confusion_matrix_at_label_key(db: Session, classification_test_data):
    label_key = "animal"
    cm = confusion_matrix_at_label_key(db, dataset_name, model_name, label_key)
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            prediction="bird", groundtruth="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="cat", groundtruth="dog", count=2
        ),
        schemas.ConfusionMatrixEntry(
            prediction="cat", groundtruth="cat", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="cat", groundtruth="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="dog", groundtruth="bird", count=1
        ),
    ]
    for entry in cm.entries:
        assert entry in expected_entries
    for entry in expected_entries:
        assert entry in cm.entries
    assert accuracy_from_cm(cm) == 2 / 6

    label_key = "color"
    cm = confusion_matrix_at_label_key(db, dataset_name, model_name, label_key)
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            prediction="white", groundtruth="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="white", groundtruth="blue", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="blue", groundtruth="white", count=1
        ),
        schemas.ConfusionMatrixEntry(
            prediction="red", groundtruth="red", count=2
        ),
        schemas.ConfusionMatrixEntry(
            prediction="red", groundtruth="black", count=1
        ),
    ]
    for entry in cm.entries:
        assert entry in expected_entries
    for entry in expected_entries:
        assert entry in cm.entries
    assert accuracy_from_cm(cm) == 3 / 6


# TODO -- Convert to use json column
# def _get_md1_val0_id(db):
#     # helper function to get metadata id for "md1", "md1-val0"
#     mds = db.scalars(select(Metadatum).where(Metadatum.key == "md1")).all()
#     md0 = mds[0]
#     assert md0.string_value == "md1-val0"

#     return md0.id


# @TODO: Will add support in second PR, need to validate `ops.Query`
# def test_confusion_matrix_at_label_key_and_group(
#     db: Session, classification_test_data  # unused except for cleanup
# ):
#     metadatum_id = _get_md1_val0_id(db)

#     cm = confusion_matrix_at_label_key(
#         db,
#         dataset=dataset_name,
#         model=model_name,
#         label_key="animal",
#         metadatum_id=metadatum_id,
#     )

#     # for this metadatum and label id we have the gts
#     # ["bird", "dog", "bird", "bird", "dog"] and the preds
#     # ["bird", "cat", "cat", "dog", "cat"]
#     expected_entries = [
#         schemas.ConfusionMatrixEntry(
#             groundtruth="bird", prediction="bird", count=1
#         ),
#         schemas.ConfusionMatrixEntry(
#             groundtruth="dog", prediction="cat", count=2
#         ),
#         schemas.ConfusionMatrixEntry(
#             groundtruth="bird", prediction="cat", count=1
#         ),
#         schemas.ConfusionMatrixEntry(
#             groundtruth="bird", prediction="dog", count=1
#         ),
#     ]

#     assert len(cm.entries) == len(expected_entries)
#     for e in expected_entries:
#         assert e in cm.entries


def test_roc_auc(db, classification_test_data):
    """Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

    ```
    from sklearn.metrics import roc_auc_score

    # for the "animal" label key
    y_true = [0, 2, 0, 0, 1, 2]
    y_score = [
        [0.6, 0.2, 0.2],
        [0.0, 0.9, 0.1],
        [0.15, 0.8, 0.05],
        [0.15, 0.1, 0.75],
        [0.0, 1.0, 0.0],
        [0.2, 0.4, 0.4],
    ]

    print(roc_auc_score(y_true, y_score, multi_class="ovr"))

    # for the "color" label key
    y_true = [3, 3, 2, 1, 0, 2]
    y_score = [
        [0.05, 0.2, 0.1, 0.65],
        [0.2, 0.5, 0.0, 0.3],
        [0.3, 0.1, 0.4, 0.2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.03, 0.01, 0.9, 0.06],
    ]
    ```

    outputs:

    ```
    0.8009259259259259
    0.43125
    ```
    """
    assert (
        roc_auc(db, dataset_name, model_name, label_key="animal")
        == 0.8009259259259259
    )
    assert roc_auc(db, dataset_name, model_name, label_key="color") == 0.43125

    with pytest.raises(RuntimeError) as exc_info:
        roc_auc(db, dataset_name, model_name, label_key="not a key")
    assert "is not a classification label" in str(exc_info)


def _create_gt_data(
    db: Session, gt_semantic_segs_create: list[schemas.GroundTruth]
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_name),
    )
    for gt in gt_semantic_segs_create:
        crud.create_groundtruth(db=db, groundtruth=gt)


def _create_data(
    db: Session,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_gt_data(db=db, gt_semantic_segs_create=gt_semantic_segs_create)

    crud.create_model(db=db, model=schemas.Model(name=model_name))

    crud.create_prediction(db=db, prediction=pred_semantic_segs_img1_create)
    crud.create_prediction(db=db, prediction=pred_semantic_segs_img2_create)


def test__gt_query_and_pred_query(
    db: Session,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    for label_key, label_value, expected_number in [
        ("k1", "v1", 2),
        ("k1", "v2", 1),
        ("k2", "v2", 1),
        ("k3", "v3", 1),
    ]:
        label_id = db.scalar(
            select(Label.id).where(
                and_(Label.key == label_key, Label.value == label_value)
            )
        )

        assert label_id is not None

        q = _gt_query(dataset_name, label_id=label_id)
        data = db.execute(q).all()
        assert len(data) == expected_number

    q = _gt_query(dataset_name, label_id=10000000)
    data = db.execute(q).all()
    assert len(data) == 0

    for label_key, label_value, expected_number in [
        ("k1", "v1", 2),
        ("k1", "v2", 0),
        ("k2", "v2", 1),
        ("k2", "v3", 2),
    ]:
        label_id = db.scalar(
            select(Label.id).where(
                and_(Label.key == label_key, Label.value == label_value)
            )
        )

        assert label_id is not None

        q = _pred_query(dataset_name, model_name=model_name, label_id=label_id)
        data = db.execute(q).all()
        assert len(data) == expected_number

    q = _pred_query(dataset_name, model_name=model_name, label_id=10000000)
    data = db.execute(q).all()
    assert len(data) == 0


def _gt_tuples(gts: list[schemas.GroundTruth], label: schemas.Label):
    return [
        (gt.datum.uid, ann.raster.array)
        for gt in gts
        for ann in gt.annotations
        if label in ann.labels
    ]


def _pred_tuples(preds: list[schemas.Prediction], label: schemas.Label):
    return [
        (pred.datum.uid, ann.raster.array)
        for pred in preds
        for ann in pred.annotations
        if label in ann.labels
    ]


def _tp_count(
    gts: list[schemas.GroundTruth],
    preds: list[schemas.Prediction],
    label: schemas.Label,
) -> int:
    gts = _gt_tuples(gts, label)
    preds = _pred_tuples(preds, label)

    datum_ids = set([gt[0] for gt in gts]).intersection(
        [pred[0] for pred in preds]
    )

    ret = 0
    for datum_id in datum_ids:
        gt_mask = [gt[1] for gt in gts if gt[0] == datum_id][0]
        pred_mask = [pred[1] for pred in preds if pred[0] == datum_id][0]

        ret += (gt_mask * pred_mask).sum()

    return ret


def test_tp_count(
    db: Session,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    for k, v in [("k1", "v1"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _tp_count(
            gt_semantic_segs_create,
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        tps = tp_count(
            db=db,
            dataset_name=dataset_name,
            model_name=model_name,
            label_id=label_id,
        )

        assert expected == tps


def _gt_count(gts: list[schemas.GroundTruth], label: schemas.Label) -> int:
    gts = _gt_tuples(gts, label)

    ret = 0
    for gt in gts:
        ret += gt[1].sum()

    return ret


def test_gt_count(
    db: Session, gt_semantic_segs_create: list[schemas.GroundTruth]
):
    _create_gt_data(db=db, gt_semantic_segs_create=gt_semantic_segs_create)

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k3", "v3"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _gt_count(
            gt_semantic_segs_create, schemas.Label(key=k, value=v)
        )

        assert (
            gt_count(db=db, dataset_name=dataset_name, label_id=label_id)
            == expected
        )

    with pytest.raises(RuntimeError) as exc_info:
        gt_count(db=db, dataset_name=dataset_name, label_id=1000000)

    assert "No groundtruth pixels for label" in str(exc_info)


def _pred_count(preds: list[schemas.Prediction], label: schemas.Label) -> int:
    preds = _pred_tuples(preds, label)

    ret = 0
    for pred in preds:
        ret += pred[1].sum()

    return ret


def test_pred_count(
    db: Session,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    for k, v in [("k1", "v1"), ("k1", "v2"), ("k2", "v3"), ("k2", "v2")]:
        label_id = db.scalar(
            select(Label).where(and_(Label.key == k, Label.value == v))
        ).id

        expected = _pred_count(
            [pred_semantic_segs_img1_create, pred_semantic_segs_img2_create],
            schemas.Label(key=k, value=v),
        )

        assert (
            pred_count(
                db=db,
                model_name=model_name,
                dataset_name=dataset_name,
                label_id=label_id,
            )
            == expected
        )

    assert (
        pred_count(
            db=db,
            dataset_name=dataset_name,
            model_name=model_name,
            label_id=1000000,
        )
        == 0
    )


def test_get_groundtruth_labels(
    db: Session, gt_semantic_segs_create: list[schemas.GroundTruth]
):
    _create_gt_data(db=db, gt_semantic_segs_create=gt_semantic_segs_create)
    labels = get_groundtruth_labels(db, dataset_name)

    assert len(labels) == 4

    assert set([label[:2] for label in labels]) == {
        ("k1", "v1"),
        ("k1", "v2"),
        ("k2", "v2"),
        ("k3", "v3"),
    }

    assert len(set([label[-1] for label in labels])) == 4


def test_compute_segmentation_metrics(
    db: Session,
    gt_semantic_segs_create: list[schemas.GroundTruth],
    pred_semantic_segs_img1_create: schemas.Prediction,
    pred_semantic_segs_img2_create: schemas.Prediction,
):
    _create_data(
        db=db,
        gt_semantic_segs_create=gt_semantic_segs_create,
        pred_semantic_segs_img1_create=pred_semantic_segs_img1_create,
        pred_semantic_segs_img2_create=pred_semantic_segs_img2_create,
    )

    metrics = compute_segmentation_metrics(db, dataset_name, model_name)
    # should have five metrics (one IOU for each of the four labels, and one mIOU)
    assert len(metrics) == 5
    for metric in metrics[:-1]:
        assert isinstance(metric, schemas.IOUMetric)
        assert metric.value < 1.0
    assert isinstance(metrics[-1], schemas.mIOUMetric)
    assert metrics[-1].value < 1.0


# @TODO: Will support in second PR, need to validate `ops.Query`
# def test_roc_auc_groupby_metadata(db, classification_test_data):
#     """Test computing ROC AUC for a given grouping. This agrees with:

#     Scikit-learn won't do multiclass ROC AUC when there are only two predictive classes. So we
#     compare this to doing the following in scikit-learn: first computing binary ROC for the "dog" class via:

#     ```
#     from sklearn.metrics import roc_auc_score

#     y_true = [0, 1, 0, 0, 1]
#     y_score = [0.2, 0.1, 0.05, 0.75, 0.4]

#     roc_auc_score(y_true, y_score)
#     ```

#     which gives 0.5. Then we do it for the "bird" class via:

#     ```
#     from sklearn.metrics import roc_auc_score

#     y_true = [1, 0, 1, 1, 0]
#     y_score = [0.6, 0.0, 0.15, 0.15, 0.2]

#     roc_auc_score(y_true, y_score)
#     ```

#     which gives 2/3. So we expect our implementation to give the average of 0.5 and 2/3
#     """

#     metadatum_id = _get_md1_val0_id(db)

#     assert (
#         roc_auc(
#             db,
#             dataset_name,
#             model_name,
#             label_key="animal",
#             metadatum_id=metadatum_id,
#         )
#         == (0.5 + 2 / 3) / 2
#     )

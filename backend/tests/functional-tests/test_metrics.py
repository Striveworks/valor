import io
from base64 import b64encode

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from velour_api import crud, models, schemas
from velour_api.metrics import compute_ap_metrics
from velour_api.metrics.classification import (
    accuracy_from_cm,
    confusion_matrix_at_label_key,
    roc_auc,
)
from velour_api.metrics.segmentation import tp_count
from velour_api.models import (
    LabeledGroundTruthDetection,
    LabeledPredictedDetection,
)

dataset_name = "test dataset"
model_name = "test model"


def bytes_to_array(b: bytes) -> np.ndarray:
    f = io.BytesIO(b)
    img = Image.open(f)
    return np.array(img)


@pytest.fixture
def classification_test_data(db: Session):
    crud.create_dataset(
        db,
        schemas.DatasetCreate(
            name=dataset_name, type=schemas.DatumTypes.IMAGE
        ),
    )
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )

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
        schemas.Image(
            uid=f"uid{i}",
            height=128,
            width=256,
            metadata=[
                schemas.DatumMetadatum(
                    name="md1", value=f"md1-val{int(i == 4)}"
                ),
                schemas.DatumMetadatum(name="md2", value=f"md1-val{i % 3}"),
            ],
        )
        for i in range(6)
    ]

    gts = [
        schemas.GroundTruthClassification(
            datum=imgs[i],
            labels=[
                schemas.Label(key="animal", value=animal_gts[i]),
                schemas.Label(key="color", value=color_gts[i]),
            ],
        )
        for i in range(6)
    ]
    preds = [
        schemas.PredictedClassification(
            datum=imgs[i],
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

    crud.create_ground_truth_classifications(
        db,
        data=schemas.GroundTruthClassificationsCreate(
            dataset_name=dataset_name, classifications=gts
        ),
    )
    crud.create_predicted_image_classifications(
        db,
        data=schemas.PredictedClassificationsCreate(
            model_name=model_name,
            dataset_name=dataset_name,
            classifications=preds,
        ),
    )


@pytest.fixture
def gt_semantic_segs_create(
    img1_gt_mask_bytes1: bytes,
    img1_gt_mask_bytes2: bytes,
    img1_gt_mask_bytes3: bytes,
    img2_gt_mask_bytes1,
    img1: schemas.Image,
    img2: schemas.Image,
) -> schemas.GroundTruthSegmentationsCreate:
    b64_mask1 = b64encode(img1_gt_mask_bytes1).decode()
    b64_mask2 = b64encode(img1_gt_mask_bytes2).decode()
    b64_mask3 = b64encode(img1_gt_mask_bytes3).decode()
    b64_mask4 = b64encode(img2_gt_mask_bytes1).decode()
    return schemas.GroundTruthSegmentationsCreate(
        dataset_name=dataset_name,
        segmentations=[
            schemas.GroundTruthSegmentation(
                is_instance=False,
                shape=b64_mask1,
                image=img1,
                labels=[schemas.Label(key="k1", value="v1")],
            ),
            schemas.GroundTruthSegmentation(
                is_instance=False,
                shape=b64_mask2,
                image=img1,
                labels=[schemas.Label(key="k1", value="v1")],
            ),
            schemas.GroundTruthSegmentation(
                is_instance=False,
                shape=b64_mask3,
                image=img1,
                labels=[schemas.Label(key="k3", value="v3")],
            ),
            schemas.GroundTruthSegmentation(
                is_instance=False,
                shape=b64_mask4,
                image=img2,
                labels=[schemas.Label(key="k1", value="v1")],
            ),
        ],
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
    model_name = "test model"
    dataset_name = "test dataset"

    dataset_id = crud.get_dataset(db, dataset_name).id
    model_id = crud.get_model(db, model_name).id

    iou_thresholds = set([round(0.5 + 0.05 * i, 2) for i in range(10)])
    metrics = compute_ap_metrics(
        db=db,
        dataset_id=dataset_id,
        model_id=model_id,
        gt_type=schemas.Task.BBOX_OBJECT_DETECTION,
        pd_type=schemas.Task.BBOX_OBJECT_DETECTION,
        label_key="class",
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


def _get_md1_val0_id(db):
    # helper function to get metadata id for "md1", "md1-val0"
    mds = db.scalars(
        select(models.Metadatum).where(models.Metadatum.name == "md1")
    ).all()
    md0 = mds[0]
    assert md0.string_value == "md1-val0"

    return md0.id


def test_confusion_matrix_at_label_key_and_group(
    db: Session, classification_test_data  # unused except for cleanup
):
    metadatum_id = _get_md1_val0_id(db)

    cm = confusion_matrix_at_label_key(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
        label_key="animal",
        metadatum_id=metadatum_id,
    )

    # for this metadatum and label id we have the gts
    # ["bird", "dog", "bird", "bird", "dog"] and the preds
    # ["bird", "cat", "cat", "dog", "cat"]
    expected_entries = [
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="bird", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="dog", prediction="cat", count=2
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="cat", count=1
        ),
        schemas.ConfusionMatrixEntry(
            groundtruth="bird", prediction="dog", count=1
        ),
    ]

    assert len(cm.entries) == len(expected_entries)
    for e in expected_entries:
        assert e in cm.entries


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


def test_roc_auc_groupby_metadata(db, classification_test_data):
    """Test computing ROC AUC for a given grouping. This agrees with:

    Scikit-learn won't do multiclass ROC AUC when there are only two predictive classes. So we
    compare this to doing the following in scikit-learn: first computing binary ROC for the "dog" class via:

    ```
    from sklearn.metrics import roc_auc_score

    y_true = [0, 1, 0, 0, 1]
    y_score = [0.2, 0.1, 0.05, 0.75, 0.4]

    roc_auc_score(y_true, y_score)
    ```

    which gives 0.5. Then we do it for the "bird" class via:

    ```
    from sklearn.metrics import roc_auc_score

    y_true = [1, 0, 1, 1, 0]
    y_score = [0.6, 0.0, 0.15, 0.15, 0.2]

    roc_auc_score(y_true, y_score)
    ```

    which gives 2/3. So we expect our implementation to give the average of 0.5 and 2/3
    """

    metadatum_id = _get_md1_val0_id(db)

    assert (
        roc_auc(
            db,
            dataset_name,
            model_name,
            label_key="animal",
            metadatum_id=metadatum_id,
        )
        == (0.5 + 2 / 3) / 2
    )


def test_tp_count(
    db: Session,
    gt_semantic_segs_create: schemas.GroundTruthSegmentationsCreate,
    pred_semantic_segs_img1_create: schemas.PredictedSegmentationsCreate,
    pred_semantic_segs_img2_create: schemas.PredictedSegmentationsCreate,
    img1_gt_mask_bytes1: bytes,
    img1_gt_mask_bytes2: bytes,
    img1_pred_mask_bytes1: bytes,
    img2_gt_mask_bytes1: bytes,
    img2_pred_mask_bytes1: bytes,
):
    crud.create_dataset(
        db,
        schemas.DatasetCreate(
            name=dataset_name, type=schemas.DatumTypes.IMAGE
        ),
    )
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    crud.create_groundtruth_segmentations(db, gt_semantic_segs_create)
    crud.create_predicted_segmentations(db, pred_semantic_segs_img1_create)
    crud.create_predicted_segmentations(db, pred_semantic_segs_img2_create)

    # preds = db.scalars(select(models.PredictedSegmentation)).all()
    # gts = db.scalars(select(models.GroundTruthSegmentation)).all()

    # pred = preds[0]
    # gt = gts[0]

    # from geoalchemy2.functions import ST_Area, ST_Count, ST_MapAlgebra

    # c = db.scalar(
    #     ST_Count(ST_MapAlgebra(gt.shape, pred.shape, "[rast1]*[rast2]"))
    # )

    # gt_mask1_arr = bytes_to_array(img1_gt_mask_bytes1)
    # pred_mask1_arr = bytes_to_array(img1_pred_mask_bytes1)
    # assert c == (gt_mask1_arr * pred_mask1_arr).sum()

    # check we get the correct tp count for the label ("k1", "v1").
    # For img1 this has groundtruth img1_gt_mask_bytes1, img1_gt_mask_bytes2 and prediction img1_pred_mask_bytes1
    # and for img2 this has groundtruth img2_gt_mask_bytes1 and prediction img2_pred_mask_bytes1

    label_id = db.scalar(
        select(models.Label).where(
            and_(models.Label.key == "k1", models.Label.value == "v1")
        )
    ).id

    tps = tp_count(db, dataset_name, model_name, label_id).all()

    # img1_gt_mask_arr1 = bytes_to_array(img1_gt_mask_bytes1)
    # img1_gt_mask_arr2 = bytes_to_array(img1_gt_mask_bytes2)
    # img1_pred_mask_arr1 = (bytes_to_array(img1_pred_mask_bytes1),)
    # img2_gt_mask_arr1 = bytes_to_array(img2_gt_mask_bytes1)
    # img2_pred_mask_arr1 = bytes_to_array(img2_pred_mask_bytes1)

    # img1_tps = (img1_gt_mask_arr1 *  )

    print(len(tps))

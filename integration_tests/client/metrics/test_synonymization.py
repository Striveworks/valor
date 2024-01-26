""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from dataclasses import asdict

import pytest
from sqlalchemy.orm import Session

from velour import Annotation, Dataset, GroundTruth, Label, Model, Prediction
from velour.client import Client
from velour.enums import EvaluationStatus, TaskType
from velour.metatypes import ImageMetadata
from velour.schemas import BoundingBox
from velour.schemas.filters import Filter

default_filter_properties = asdict(Filter())


@pytest.fixture
def gts_det_syn(
    rect1: BoundingBox,
    rect2: BoundingBox,
    rect3: BoundingBox,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="class_name", value="maine coon cat")],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="class", value="british shorthair")],
                    bounding_box=rect3,
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=rect3,
                ),
            ],
        ),
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="class", value="siamese cat")],
                    bounding_box=rect2,
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=rect2,
                ),
            ],
        ),
    ]


@pytest.fixture
def preds_det_syn(
    rect1: BoundingBox,
    rect2: BoundingBox,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[Prediction]:
    return [
        Prediction(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="class", value="cat", score=0.3)],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1", score=0.3)],
                    bounding_box=rect1,
                ),
            ],
        ),
        Prediction(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="class_name", value="cat", score=0.98)],
                    bounding_box=rect2,
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k2", value="v2", score=0.98)],
                    bounding_box=rect2,
                ),
            ],
        ),
    ]


def test_detection_synonymization(
    db: Session,
    dataset_name: str,
    model_name: str,
    client: Client,
    gts_det_syn: list[GroundTruth],
    preds_det_syn: list[Prediction],
):
    dataset = Dataset(client, dataset_name)

    for gt in gts_det_syn:
        dataset.add_groundtruth(gt)

    dataset.finalize()

    model = Model(client, model_name)

    for pd in preds_det_syn:
        model.add_prediction(dataset, pd)

    model.finalize_inferences(dataset)

    # for the first evaluation, don't do anything about the mismatched labels
    # we expect the evaluation to return the same expected metrics as for our standard detection tests
    baseline_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1},
            "value": 0.07213578500707214,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6},
            "value": 0.07213578500707214,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.07213578500707214,
        },
    ]

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
    )

    assert (
        len(eval_job.ignored_pred_labels) == 2
    )  # we're ignoring the two "cat" model predictions
    assert (
        len(eval_job.missing_pred_labels) == 3
    )  # we're missing three gts_det_syn representing different breeds of cats

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics
    for m in metrics:
        assert m in baseline_expected_metrics
    for m in baseline_expected_metrics:
        assert m in metrics

    # now, we correct most of the mismatched labels with a label map
    cat_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1},
            "value": 0.21039603960396042,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6},
            "value": 0.21039603960396042,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.21039603960396042,
        },
    ]

    label_mapping = {
        Label(key="class_name", value="maine coon cat"): Label(
            key="class", value="cat"
        ),
        Label(key="class", value="siamese cat"): Label(
            key="class", value="cat"
        ),
        Label(key="class", value="british shorthair"): Label(
            key="class", value="cat"
        ),
    }

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        label_map=label_mapping,
    )

    assert (
        len(eval_job.ignored_pred_labels) == 1
    )  # Label(key='class_name', value='cat', score=None) is still never used
    assert len(eval_job.missing_pred_labels) == 0

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics
    for m in metrics:
        assert m in cat_expected_metrics
    for m in cat_expected_metrics:
        assert m in metrics

    assert eval_job.parameters.label_map == [
        [["class_name", "maine coon cat"], ["class", "cat"]],
        [["class", "siamese cat"], ["class", "cat"]],
        [["class", "british shorthair"], ["class", "cat"]],
    ]

    # finally, we check that the label mapping works when the label is completely foreign
    # to both groundtruths and predictions
    foo_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1},
            "value": 0.3894389438943895,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6},
            "value": 0.3894389438943895,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.3894389438943895,
        },
    ]

    label_mapping = {
        # map the groundtruths
        Label(key="class_name", value="maine coon cat"): Label(
            key="foo", value="bar"
        ),
        Label(key="class", value="siamese cat"): Label(key="foo", value="bar"),
        Label(key="class", value="british shorthair"): Label(
            key="foo", value="bar"
        ),
        # map the predictions
        Label(key="class", value="cat"): Label(key="foo", value="bar"),
        Label(key="class_name", value="cat"): Label(key="foo", value="bar"),
    }

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        label_map=label_mapping,
    )

    assert len(eval_job.ignored_pred_labels) == 0
    assert len(eval_job.missing_pred_labels) == 0
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics
    for m in metrics:
        assert m in foo_expected_metrics
    for m in foo_expected_metrics:
        assert m in metrics

    assert eval_job.parameters.label_map == [
        [["class_name", "maine coon cat"], ["foo", "bar"]],
        [["class", "siamese cat"], ["foo", "bar"]],
        [["class", "british shorthair"], ["foo", "bar"]],
        [["class", "cat"], ["foo", "bar"]],
        [["class_name", "cat"], ["foo", "bar"]],
    ]


@pytest.fixture
def gt_clfs_syn(
    img5: ImageMetadata,
    img6: ImageMetadata,
    img8: ImageMetadata,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img5.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4"),
                        Label(key="k5", value="v5"),
                        Label(key="class", value="siamese cat"),
                    ],
                ),
            ],
        ),
        GroundTruth(
            datum=img6.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4"),
                        Label(key="class", value="british shorthair"),
                    ],
                )
            ],
        ),
        GroundTruth(
            datum=img8.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k3", value="v3"),
                        Label(key="class", value="tabby cat"),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_syn(
    model_name: str, img5: ImageMetadata, img6: ImageMetadata
) -> list[Prediction]:
    return [
        Prediction(
            datum=img5.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k12", value="v12", score=0.47),
                        Label(key="k12", value="v16", score=0.53),
                        Label(key="k13", value="v13", score=1.0),
                        Label(key="class", value="cat", score=1),
                    ],
                )
            ],
        ),
        Prediction(
            datum=img6.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4", score=0.71),
                        Label(key="k4", value="v5", score=0.29),
                        Label(key="class_name", value="cat", score=1),
                    ],
                )
            ],
        ),
    ]


def test_classification_synonymization(
    client: Client,
    gt_clfs_syn: list[GroundTruth],
    pred_clfs_syn: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset(client, dataset_name)
    for gt in gt_clfs_syn:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model(client, model_name)
    for pd in pred_clfs_syn:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    # check baseline case

    baseline_expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 1.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k4"}, "value": 1.0},
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v5"}},
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {"type": "F1", "value": 1.0, "label": {"key": "k4", "value": "v4"}},
        {
            "type": "Accuracy",
            "parameters": {"label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "class"},
            "value": -1.0,
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "F1",
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "F1",
            "value": 0.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "class", "value": "tabby cat"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "class", "value": "tabby cat"},
        },
        {
            "type": "F1",
            "value": -1.0,
            "label": {"key": "class", "value": "tabby cat"},
        },
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "F1",
            "value": -1.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
    ]

    baseline_cm = [
        {
            "label_key": "class",
            "entries": [
                {"prediction": "cat", "groundtruth": "siamese cat", "count": 1}
            ],
        },
        {
            "label_key": "k4",
            "entries": [{"prediction": "v4", "groundtruth": "v4", "count": 1}],
        },
    ]

    eval_job = model.evaluate_classification(dataset)

    assert eval_job.id
    assert set(eval_job.ignored_pred_keys) == {"class_name", "k12", "k13"}
    assert set(eval_job.missing_pred_keys) == {"k3", "k5"}

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    for m in metrics:
        assert m in baseline_expected_metrics
    for m in baseline_expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.confusion_matrices
    assert all([entry in baseline_cm for entry in confusion_matrices])

    # now try using a label map to connect all the cats

    label_mapping = {
        # map the groundtruths
        Label(key="class", value="tabby cat"): Label(
            key="special_class", value="cat_type1"
        ),
        Label(key="class", value="siamese cat"): Label(
            key="special_class", value="cat_type1"
        ),
        Label(key="class", value="british shorthair"): Label(
            key="special_class", value="cat_type2"
        ),
        # map the predictions
        Label(key="class", value="cat"): Label(
            key="special_class", value="cat_type1"
        ),
        Label(key="class_name", value="cat"): Label(
            key="special_class", value="cat_type1"
        ),
    }

    cat_expected_metrics = [
        {
            "type": "Accuracy",
            "parameters": {"label_key": "special_class"},
            "value": 0.5,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "special_class"},
            "value": -1.0,
        },
        {
            "type": "Precision",
            "value": 0.5,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "special_class", "value": "cat_type2"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "special_class", "value": "cat_type2"},
        },
        {
            "type": "F1",
            "value": 0.0,
            "label": {"key": "special_class", "value": "cat_type2"},
        },
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 1.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k4"}, "value": 1.0},
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v5"}},
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {"type": "F1", "value": 1.0, "label": {"key": "k4", "value": "v4"}},
    ]

    cat_expected_cm = [
        {
            "label_key": "special_class",
            "entries": [
                {
                    "prediction": "cat_type1",
                    "groundtruth": "cat_type1",
                    "count": 1,
                },
                {
                    "prediction": "cat_type1",
                    "groundtruth": "cat_type2",
                    "count": 1,
                },
            ],
        },
        {
            "label_key": "k4",
            "entries": [{"prediction": "v4", "groundtruth": "v4", "count": 1}],
        },
    ]

    eval_job = model.evaluate_classification(dataset, label_map=label_mapping)

    assert eval_job.id
    assert set(eval_job.ignored_pred_keys) == {"k12", "k13"}
    assert set(eval_job.missing_pred_keys) == {"k3", "k5"}

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics
    for m in metrics:
        assert m in cat_expected_metrics
    for m in cat_expected_metrics:
        assert m in metrics

    confusion_matrix = eval_job.confusion_matrices

    for row in confusion_matrix:
        if row["label_key"] == "special_class":
            for entry in cat_expected_cm[0]["entries"]:
                assert entry in row["entries"]
            for entry in row["entries"]:
                assert entry in cat_expected_cm[0]["entries"]
        else:  # check k4, v4 entry
            for entry in cat_expected_cm[1]["entries"]:
                assert entry in row["entries"]
            for entry in row["entries"]:
                assert entry in cat_expected_cm[1]["entries"]


def test_evaluate_segmentation_synonmization(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
):
    dataset = Dataset(client, dataset_name)
    model = Model(client, model_name)

    for gt in gt_semantic_segs1:
        gt.datum.metadata["color"] = "red"
        dataset.add_groundtruth(gt)
    for gt in gt_semantic_segs2:
        gt.datum.metadata["color"] = "blue"
        dataset.add_groundtruth(gt)

    dataset.finalize()

    for pred in pred_semantic_segs:
        model.add_prediction(dataset, pred)

    model.finalize_inferences(dataset)

    # check the baseline case

    eval_job = model.evaluate_segmentation(dataset)
    assert eval_job.missing_pred_labels == [
        {"key": "k3", "value": "v3", "score": None}
    ]
    assert eval_job.ignored_pred_labels == [
        {"key": "k1", "value": "v1", "score": None}
    ]
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    assert len(metrics) == 3
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("k2", "v2"), ("k3", "v3")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}

    # now do the same thing, but with a label map

    eval_job = model.evaluate_segmentation(
        dataset,
        label_map={
            Label(key=f"k{i}", value=f"v{i}"): Label(key="foo", value="bar")
            for i in range(1, 4)
        },
    )

    # no labels are missing, since the missing labels have been mapped to a grouper label
    assert eval_job.missing_pred_labels == []
    assert eval_job.ignored_pred_labels == []
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    # there's now only two metrics, since all three (k, v) combinations have been mapped to (foo, bar)
    assert len(metrics) == 2
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("foo", "bar"), ("foo", "bar")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}

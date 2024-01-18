""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from dataclasses import asdict

import pytest
from sqlalchemy.orm import Session

from velour import Annotation, Dataset, GroundTruth, Label, Model, Prediction
from velour.client import Client
from velour.enums import TaskType
from velour.metatypes import ImageMetadata
from velour.schemas import BoundingBox
from velour.schemas.filters import Filter

default_filter_properties = asdict(Filter())


@pytest.fixture
def gts(
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
def preds(
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


@pytest.fixture
def expected_metrics():
    return [
        {
            "type": "AP",
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
            "parameters": {
                "iou": 0.1,
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
            "parameters": {
                "iou": 0.6,
            },
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
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
            "value": 0.504950495049505,
        },
    ]


def test_detection_synonymization(
    db: Session,
    dataset_name: str,
    model_name: str,
    client: Client,
    gts: list[GroundTruth],
    preds: list[Prediction],
    expected_metrics: dict,
):
    dataset = Dataset(client, dataset_name)

    for gt in gts:
        dataset.add_groundtruth(gt)

    dataset.finalize()

    model = Model(client, model_name)

    for pd in preds:
        model.add_prediction(dataset, pd)

    model.finalize_inferences(dataset)

    # for the first evaluation, don't do anything about the mismatched labels
    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
    )

    assert isinstance(eval_job.evaluation_id, int)
    assert (
        len(eval_job.ignored_pred_labels) == 2
    )  # we're ignoring the two "cat" model predictions
    assert (
        len(eval_job.missing_pred_labels) == 3
    )  # we're missing three gts representing different breeds of cats

    eval_results = eval_job.wait_for_completion()

    result = asdict(eval_results)
    assert result["metrics"] == expected_metrics

    # now, we correct the mismatched labels with a label map
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
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        label_map=label_mapping,
    )

    assert isinstance(eval_job.evaluation_id, int)
    assert len(eval_job.ignored_pred_labels) == 0
    assert len(eval_job.missing_pred_labels) == 0
    eval_results = eval_job.wait_for_completion()

    result = asdict(eval_results)

    # TODO expected metrics should be updated here
    assert result["metrics"] == expected_metrics

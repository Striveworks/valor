""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from dataclasses import asdict

import pytest
from geoalchemy2.functions import ST_Area
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour import (
    Annotation,
    Client,
    Dataset,
    Filter,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from velour.enums import AnnotationType, EvaluationStatus, TaskType
from velour.exceptions import ClientException
from velour.metatypes import ImageMetadata
from velour.schemas import BoundingBox
from velour_api.backend import models

default_filter_properties = asdict(Filter())


def test_evaluate_detection(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    """
    Test detection evaluations with area thresholds.

    gt_dets1
        datum 1
            - Label (k1, v1) with Annotation area = 1500
            - Label (k2, v2) with Annotation area = 57,510
        datum2
            - Label (k1, v1) with Annotation area = 1100

    pred_dets
        datum 1
            - Label (k1, v1) with Annotation area = 1500
            - Label (k2, v2) with Annotation area = 57,510
        datum2
            - Label (k1, v1) with Annotation area = 1100
    """
    dataset = Dataset.create(dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    for pd in pred_dets:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    expected_metrics = [
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

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    assert isinstance(eval_job.id, int)
    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []
    assert eval_job.status == EvaluationStatus.DONE

    result = eval_job
    assert result.to_dict() == {
        "id": eval_job.id,
        "model_name": model_name,
        "datum_filter": {
            **default_filter_properties,
            "dataset_names": ["test_dataset"],
            "label_keys": ["k1"],
        },
        "parameters": {
            "task_type": TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": AnnotationType.BOX.value,
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "label_map": None,
        },
        "status": EvaluationStatus.DONE.value,
        "metrics": expected_metrics,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }

    # test evaluating a job using a `Label.labels` filter
    eval_job_value_filter_using_in_ = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Annotation.labels.in_([Label(key="k1", value="v1")]),
            Annotation.bounding_box.exists(),
        ],
    )
    assert (
        eval_job_value_filter_using_in_.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    assert eval_job_value_filter_using_in_.metrics == result.metrics

    # same as the above, but not using the in_ operator
    eval_job_value_filter = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Annotation.labels == Label(key="k1", value="v1"),
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert (
        eval_job_value_filter.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    assert eval_job_value_filter.metrics == result.metrics

    # assert that this evaluation returns no metrics as there aren't any
    # Labels with key=k1 and value=v2
    with pytest.raises(ClientException) as e:
        model.evaluate_detection(
            dataset,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_return=[0.1, 0.6],
            filter_by=[
                Annotation.labels.in_([Label(key="k1", value="v2")]),
            ],
            convert_annotations_to_type=AnnotationType.BOX,
        )
    assert "EvaluationRequestError" in str(e)

    # now test if we set min_area and/or max_area
    areas = db.scalars(
        select(ST_Area(models.Annotation.box)).where(
            models.Annotation.model_id.isnot(None)
        )
    ).all()
    assert sorted(areas) == [1100.0, 1500.0]

    # sanity check this should give us the same thing except min_area and max_area are not none
    eval_job_bounded_area_10_2000 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.area >= 10,
            Annotation.bounding_box.area <= 2000,
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )

    assert (
        eval_job_bounded_area_10_2000.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    assert eval_job_bounded_area_10_2000.to_dict() == {
        "id": eval_job_bounded_area_10_2000.id,
        "model_name": model_name,
        "datum_filter": {
            **default_filter_properties,
            "dataset_names": ["test_dataset"],
            "bounding_box_area": [
                {
                    "operator": ">=",
                    "value": 10.0,
                },
                {
                    "operator": "<=",
                    "value": 2000.0,
                },
            ],
            "label_keys": ["k1"],
        },
        "parameters": {
            "task_type": TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": AnnotationType.BOX.value,
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "label_map": None,
        },
        "status": EvaluationStatus.DONE.value,
        "metrics": expected_metrics,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }

    # now check we get different things by setting the thresholds accordingly
    # min area threshold should divide the set of annotations
    eval_job_min_area_1200 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.area >= 1200,
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert (
        eval_job_min_area_1200.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    result = eval_job_min_area_1200.to_dict()
    min_area_1200_metrics = result.pop("metrics")
    assert result == {
        "id": eval_job_min_area_1200.id,
        "model_name": model_name,
        "datum_filter": {
            **default_filter_properties,
            "dataset_names": ["test_dataset"],
            "bounding_box_area": [
                {
                    "operator": ">=",
                    "value": 1200.0,
                },
            ],
            "label_keys": ["k1"],
        },
        "parameters": {
            "task_type": TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": AnnotationType.BOX.value,
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "label_map": None,
        },
        # check metrics below
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }
    assert min_area_1200_metrics != expected_metrics

    # check for difference with max area now dividing the set of annotations
    eval_job_max_area_1200 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.area <= 1200,
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert (
        eval_job_max_area_1200.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    result = eval_job_max_area_1200.to_dict()
    max_area_1200_metrics = result.pop("metrics")
    assert result == {
        "id": eval_job_max_area_1200.id,
        "model_name": model_name,
        "datum_filter": {
            **default_filter_properties,
            "dataset_names": ["test_dataset"],
            "bounding_box_area": [
                {
                    "operator": "<=",
                    "value": 1200.0,
                },
            ],
            "label_keys": ["k1"],
        },
        "parameters": {
            "task_type": TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": AnnotationType.BOX.value,
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "label_map": None,
        },
        # check metrics below
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [{"key": "k1", "value": "v1"}],
        "ignored_pred_labels": [],
    }
    assert max_area_1200_metrics != expected_metrics

    # should perform the same as the first min area evaluation
    # except now has an upper bound
    eval_job_bounded_area_1200_1800 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.area >= 1200,
            Annotation.bounding_box.area <= 1800,
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert (
        eval_job_bounded_area_1200_1800.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    result = eval_job_bounded_area_1200_1800.to_dict()
    bounded_area_metrics = result.pop("metrics")
    assert result == {
        "id": eval_job_bounded_area_1200_1800.id,
        "model_name": model_name,
        "datum_filter": {
            **default_filter_properties,
            "dataset_names": ["test_dataset"],
            "bounding_box_area": [
                {
                    "operator": ">=",
                    "value": 1200.0,
                },
                {
                    "operator": "<=",
                    "value": 1800.0,
                },
            ],
            "label_keys": ["k1"],
        },
        "parameters": {
            "task_type": TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": AnnotationType.BOX.value,
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "label_map": None,
        },
        # check metrics below
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }
    assert bounded_area_metrics != expected_metrics
    assert bounded_area_metrics == min_area_1200_metrics

    # test accessing these evaluations via the dataset
    all_evals = dataset.get_evaluations()
    assert len(all_evals) == 7


def test_evaluate_detection_with_json_filters(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    dataset = Dataset.create(dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    for pd in pred_dets:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    # test default iou arguments
    eval_results = model.evaluate_detection(
        dataset,
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.exists(),
        ],
    )
    assert (
        eval_results.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    )
    assert eval_results.parameters.iou_thresholds_to_compute == [
        i / 100 for i in range(50, 100, 5)
    ]
    assert eval_results.parameters.iou_thresholds_to_return == [
        0.5,
        0.75,
    ]

    expected_metrics = [
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

    eval_results_min_area_1200 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.area >= 1200,
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert (
        eval_results_min_area_1200.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    min_area_1200_metrics = eval_results_min_area_1200.to_dict()["metrics"]

    eval_job_bounded_area_1200_1800 = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by={
            **default_filter_properties,
            "bounding_box_area": [
                {
                    "operator": ">=",
                    "value": 1200.0,
                },
                {
                    "operator": "<=",
                    "value": 1800.0,
                },
            ],
            "label_keys": ["k1"],
        },
        convert_annotations_to_type=AnnotationType.BOX,
    )

    assert (
        eval_job_bounded_area_1200_1800.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )
    result = eval_job_bounded_area_1200_1800.to_dict()

    bounded_area_metrics = result.pop("metrics")
    assert result == {
        "id": eval_job_bounded_area_1200_1800.id,
        "model_name": model_name,
        "datum_filter": {
            **default_filter_properties,
            "dataset_names": ["test_dataset"],
            "bounding_box_area": [
                {
                    "operator": ">=",
                    "value": 1200.0,
                },
                {
                    "operator": "<=",
                    "value": 1800.0,
                },
            ],
            "label_keys": ["k1"],
        },
        "parameters": {
            "task_type": TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": AnnotationType.BOX.value,
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "label_map": None,
        },
        # check metrics below
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }
    assert bounded_area_metrics != expected_metrics
    assert bounded_area_metrics == min_area_1200_metrics


def test_get_evaluations(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    pred_dets2: list[Prediction],
):
    dataset_ = dataset_name
    model_ = model_name

    dataset = Dataset.create(dataset_)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_)
    for pd in pred_dets:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.exists(),
        ],
    )
    eval_job.wait_for_completion(timeout=30)

    expected_metrics = [
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

    second_model_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {"type": "mAP", "parameters": {"iou": 0.1}, "value": 0.0},
        {"type": "mAP", "parameters": {"iou": 0.6}, "value": 0.0},
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
        },
    ]

    # test error when we don't pass either a model or dataset
    with pytest.raises(ValueError):
        client.get_evaluations()

    evaluations = client.get_evaluations(
        datasets=dataset_name, models=model_name
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].metrics)
    assert evaluations[0].metrics == expected_metrics

    evaluations_by_evaluation_id = client.get_evaluations(
        evaluation_ids=eval_job.id
    )
    assert len(evaluations_by_evaluation_id) == 1
    assert (
        evaluations_by_evaluation_id[0].to_dict() == evaluations[0].to_dict()
    )

    # test incorrect names
    assert len(client.get_evaluations(datasets="wrong_dataset_name")) == 0
    assert len(client.get_evaluations(models="wrong_model_name")) == 0

    # test with multiple models
    second_model = Model.create("second_model")
    for pd in pred_dets2:
        second_model.add_prediction(dataset, pd)
    second_model.finalize_inferences(dataset)

    eval_job2 = second_model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Label.key == "k1",
            Annotation.bounding_box.exists(),
        ],
    )
    eval_job2.wait_for_completion(timeout=30)

    second_model_evaluations = client.get_evaluations(models="second_model")

    assert len(second_model_evaluations) == 1
    assert second_model_evaluations[0].metrics == second_model_expected_metrics

    both_evaluations = client.get_evaluations(datasets=["test_dataset"])

    # should contain two different entries, one for each model
    assert len(both_evaluations) == 2
    for evaluation in both_evaluations:
        assert evaluation.model_name in [
            "second_model",
            model_name,
        ]
        if evaluation.model_name == model_name:
            assert evaluation.metrics == expected_metrics
        elif evaluation.model_name == "second_model":
            assert evaluation.metrics == second_model_expected_metrics

    # should be equivalent since there are only two models attributed to this dataset
    both_evaluations_from_model_names = client.get_evaluations(
        models=["second_model", "test_model"]
    )
    assert len(both_evaluations_from_model_names) == 2
    assert {both_evaluations[0].id, both_evaluations[1].id} == {
        eval_.id for eval_ in both_evaluations_from_model_names
    }

    # should also be equivalent
    both_evaluations_from_evaluation_ids = client.get_evaluations(
        evaluation_ids=[eval_job.id, eval_job2.id]
    )
    assert len(both_evaluations_from_evaluation_ids) == 2
    assert {both_evaluations[0].id, both_evaluations[1].id} == {
        eval_.id for eval_ in both_evaluations_from_evaluation_ids
    }


@pytest.fixture
def gts_det_with_label_maps(
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
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="class_name", value="maine coon cat")],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="class", value="british shorthair")],
                    bounding_box=rect3,
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=rect3,
                ),
            ],
        ),
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="class", value="siamese cat")],
                    bounding_box=rect2,
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=rect2,
                ),
            ],
        ),
    ]


@pytest.fixture
def preds_det_with_label_maps(
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
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="class", value="cat", score=0.3)],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1", score=0.3)],
                    bounding_box=rect1,
                ),
            ],
        ),
        Prediction(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="class_name", value="cat", score=0.98)],
                    bounding_box=rect2,
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k2", value="v2", score=0.98)],
                    bounding_box=rect2,
                ),
            ],
        ),
    ]


def test_evaluate_detection_with_label_maps(
    db: Session,
    dataset_name: str,
    model_name: str,
    client: Client,
    gts_det_with_label_maps: list[GroundTruth],
    preds_det_with_label_maps: list[Prediction],
):
    dataset = Dataset.create(dataset_name)

    for gt in gts_det_with_label_maps:
        dataset.add_groundtruth(gt)

    dataset.finalize()

    model = Model.create(model_name)

    for pd in preds_det_with_label_maps:
        model.add_prediction(dataset, pd)

    model.finalize_inferences(dataset)

    # for the first evaluation, don't do anything about the mismatched labels
    # we expect the evaluation to return the same expected metrics as for our standard detection tests

    baseline_expected_metrics = [
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
            "label": {"key": "class", "value": "british shorthair"},
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
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
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
            "label": {"key": "class", "value": "british shorthair"},
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
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1},
            "value": 0.100990099009901,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6},
            "value": 0.100990099009901,
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
            "label": {"key": "class", "value": "british shorthair"},
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
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.100990099009901,
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
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
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
            "label": {"key": "k2", "value": "v2"},
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
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1},
            "value": 0.28052805280528054,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6},
            "value": 0.28052805280528054,
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
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.28052805280528054,
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

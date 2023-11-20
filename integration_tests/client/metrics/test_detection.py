""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from dataclasses import asdict

import pytest
from geoalchemy2.functions import ST_Area
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour import Annotation, Dataset, GroundTruth, Label, Model, Prediction
from velour.client import Client
from velour.enums import AnnotationType, JobStatus, TaskType
from velour_api.backend import models


def test_evaluate_detection(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    db: Session,
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(client, dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
    for pd in pred_dets:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
        timeout=30,
    )

    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []
    assert isinstance(eval_job._id, int)

    eval_job.wait_for_completion()
    assert eval_job.status == JobStatus.DONE

    settings = asdict(eval_job.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "settings": {
            "task_type": TaskType.DETECTION.value,
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "filters": {
                "annotation_types": ["box"],
                "label_keys": ["k1"],
            },
        },
    }

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

    assert eval_job.metrics["metrics"] == expected_metrics

    # now test if we set min_area and/or max_area
    areas = db.scalars(
        select(ST_Area(models.Annotation.box)).where(
            models.Annotation.model_id.isnot(None)
        )
    ).all()
    assert sorted(areas) == [1100.0, 1500.0]

    # sanity check this should give us the same thing except min_area and max_area are not none
    eval_job_bounded_area_10_2000 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 10,
            Annotation.geometric_area <= 2000,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_bounded_area_10_2000.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
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
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "task_type": "object-detection",
        },
    }
    assert eval_job_bounded_area_10_2000.metrics["metrics"] == expected_metrics

    # now check we get different things by setting the thresholds accordingly
    # min area threshold should divide the set of annotations
    eval_job_min_area_1200 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 1200,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_min_area_1200.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
                    {
                        "operator": ">=",
                        "value": 1200.0,
                    },
                ],
                "label_keys": ["k1"],
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "task_type": "object-detection",
        },
    }
    assert eval_job_min_area_1200.metrics["metrics"] != expected_metrics

    # check for difference with max area now dividing the set of annotations
    eval_job_max_area_1200 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area <= 1200,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_max_area_1200.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
                    {
                        "operator": "<=",
                        "value": 1200.0,
                    },
                ],
                "label_keys": ["k1"],
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "task_type": "object-detection",
        },
    }
    assert eval_job_max_area_1200.metrics["metrics"] != expected_metrics

    # should perform the same as the first min area evaluation
    # except now has an upper bound
    eval_job_bounded_area_1200_1800 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 1200,
            Annotation.geometric_area <= 1800,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_bounded_area_1200_1800.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
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
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "task_type": "object-detection",
        },
    }
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"] != expected_metrics
    )
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"]
        == eval_job_min_area_1200.metrics["metrics"]
    )


def test_evaluate_detection_with_json_filters(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    db: Session,
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(client, dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
    for pd in pred_dets:
        model.add_prediction(pd)
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

    eval_job_min_area_1200 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 1200,
        ],
        timeout=30,
    )

    eval_job_bounded_area_1200_1800 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            "annotation_types": ["box"],
            "annotation_geometric_area": [
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
        timeout=30,
    )

    settings = asdict(eval_job_bounded_area_1200_1800.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
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
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "task_type": "object-detection",
        },
    }
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"] != expected_metrics
    )
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"]
        == eval_job_min_area_1200.metrics["metrics"]
    )


def test_get_bulk_evaluations(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    pred_dets2: list[Prediction],
    db: Session,
    dataset_name: str,
    model_name: str,
):
    dataset_ = dataset_name
    model_ = model_name

    dataset = Dataset.create(client, dataset_)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_)
    for pd in pred_dets:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
        timeout=30,
    )
    eval_job.wait_for_completion()

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

    evaluations = client.get_bulk_evaluations(
        datasets=dataset_name, models=model_name
    )

    assert len(evaluations) == 1
    assert all(
        [
            name
            in [
                "dataset",
                "model",
                "settings",
                "job_id",
                "status",
                "metrics",
                "confusion_matrices",
            ]
            for evaluation in evaluations
            for name in evaluation.keys()
        ]
    )

    assert len(evaluations[0]["metrics"])
    assert evaluations[0]["metrics"] == expected_metrics

    # test incorrect names
    with pytest.raises(Exception):
        client.get_bulk_evaluations(datasets="wrong_dataset_name")

    with pytest.raises(Exception):
        client.get_bulk_evaluations(datasets="wrong_model_name")

    # test with multiple models
    second_model = Model.create(client, "second_model")
    for pd in pred_dets2:
        second_model.add_prediction(pd)
    second_model.finalize_inferences(dataset)

    eval_job = second_model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
        timeout=30,
    )
    eval_job.wait_for_completion()

    second_model_evaluations = client.get_bulk_evaluations(
        models="second_model"
    )

    assert len(second_model_evaluations) == 1
    assert all(
        [
            name
            in [
                "dataset",
                "model",
                "settings",
                "job_id",
                "status",
                "metrics",
                "confusion_matrices",
            ]
            for evaluation in second_model_evaluations
            for name in evaluation.keys()
        ]
    )
    assert (
        second_model_evaluations[0]["metrics"] == second_model_expected_metrics
    )

    both_evaluations = client.get_bulk_evaluations(datasets=["test_dataset"])

    # should contain two different entries, one for each model
    assert len(both_evaluations) == 2
    assert all(
        [
            evaluation["model"] in ["second_model", model_name]
            for evaluation in both_evaluations
        ]
    )
    assert both_evaluations[0]["metrics"] == expected_metrics
    assert both_evaluations[1]["metrics"] == second_model_expected_metrics

    # should be equivalent since there are only two models attributed to this dataset
    both_evaluations_from_model_names = client.get_bulk_evaluations(
        models=["second_model", "test_model"]
    )
    assert both_evaluations == both_evaluations_from_model_names

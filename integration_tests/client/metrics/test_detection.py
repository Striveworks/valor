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
from velour.enums import AnnotationType, JobStatus
from velour_api.backend import models


def test_evaluate_detection(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    dataset = Dataset(client, dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model(client, model_name)
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
    assert isinstance(eval_job.evaluation_id, int)
    assert eval_job.task_type == "object-detection"
    assert eval_job.status.value == "done"
    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []

    eval_job.wait_for_completion()
    assert eval_job.status == JobStatus.DONE

    # test get_evaluation_status
    assert (
        client.get_evaluation_status(eval_job.evaluation_id) == eval_job.status
    )

    settings = asdict(eval_job.job_request)
    settings.pop("id")
    assert settings == {
        "model": "test_model",
        "dataset": "test_dataset",
        "task_type": "object-detection",
        "settings": {
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "filters": {
                "dataset_names": None,
                "dataset_metadata": None,
                "dataset_geospatial": None,
                "models_names": None,
                "models_metadata": None,
                "models_geospatial": None,
                "datum_uids": None,
                "datum_metadata": None,
                "datum_geospatial": None,
                "task_types": None,
                "annotation_types": ["box"],
                "annotation_geometric_area": None,
                "annotation_metadata": None,
                "annotation_geospatial": None,
                "prediction_scores": None,
                "labels": None,
                "label_ids": None,
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

    assert eval_job.results().metrics == expected_metrics

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
    job_request = asdict(eval_job_bounded_area_10_2000.job_request)
    job_request.pop("id")
    assert job_request == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
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
                "annotation_geospatial": None,
                "annotation_metadata": None,
                "dataset_geospatial": None,
                "dataset_metadata": None,
                "dataset_names": None,
                "datum_geospatial": None,
                "datum_metadata": None,
                "datum_uids": None,
                "label_ids": None,
                "labels": None,
                "models_geospatial": None,
                "models_metadata": None,
                "models_names": None,
                "prediction_scores": None,
                "task_types": None,
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert eval_job_bounded_area_10_2000.results().metrics == expected_metrics

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
    job_request = asdict(eval_job_min_area_1200.job_request)
    job_request.pop("id")
    assert job_request == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
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
                "annotation_geospatial": None,
                "annotation_metadata": None,
                "dataset_geospatial": None,
                "dataset_metadata": None,
                "dataset_names": None,
                "datum_geospatial": None,
                "datum_metadata": None,
                "datum_uids": None,
                "label_ids": None,
                "labels": None,
                "models_geospatial": None,
                "models_metadata": None,
                "models_names": None,
                "prediction_scores": None,
                "task_types": None,
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert eval_job_min_area_1200.results().metrics != expected_metrics

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
    job_request = asdict(eval_job_max_area_1200.job_request)
    job_request.pop("id")
    assert job_request == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
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
                "annotation_geospatial": None,
                "annotation_metadata": None,
                "dataset_geospatial": None,
                "dataset_metadata": None,
                "dataset_names": None,
                "datum_geospatial": None,
                "datum_metadata": None,
                "datum_uids": None,
                "label_ids": None,
                "labels": None,
                "models_geospatial": None,
                "models_metadata": None,
                "models_names": None,
                "prediction_scores": None,
                "task_types": None,
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert eval_job_max_area_1200.results().metrics != expected_metrics

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
    job_request = asdict(eval_job_bounded_area_1200_1800.job_request)
    job_request.pop("id")
    assert job_request == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
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
                "annotation_geospatial": None,
                "annotation_metadata": None,
                "dataset_geospatial": None,
                "dataset_metadata": None,
                "dataset_names": None,
                "datum_geospatial": None,
                "datum_metadata": None,
                "datum_uids": None,
                "label_ids": None,
                "labels": None,
                "models_geospatial": None,
                "models_metadata": None,
                "models_names": None,
                "prediction_scores": None,
                "task_types": None,
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert eval_job_bounded_area_1200_1800.results().metrics != expected_metrics
    assert (
        eval_job_bounded_area_1200_1800.results().metrics
        == eval_job_min_area_1200.results().metrics
    )

    # test accessing these evaluations via the dataset
    all_evals = dataset.get_evaluations()
    assert len(all_evals) == 5


def test_evaluate_detection_with_json_filters(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    dataset = Dataset(client, dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model(client, model_name)
    for pd in pred_dets:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    # test default iou arguments
    eval_job = model.evaluate_detection(
        dataset=dataset,
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
        timeout=30,
    )
    assert eval_job.settings.parameters.iou_thresholds_to_compute == [
        i / 100 for i in range(50, 100, 5)
    ]
    assert eval_job.settings.parameters.iou_thresholds_to_keep == [0.5, 0.75]

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

    job_request = asdict(eval_job_bounded_area_1200_1800.job_request)
    job_request.pop("id")
    assert job_request == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
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
                "annotation_geospatial": None,
                "annotation_metadata": None,
                "dataset_geospatial": None,
                "dataset_metadata": None,
                "dataset_names": None,
                "datum_geospatial": None,
                "datum_metadata": None,
                "datum_uids": None,
                "label_ids": None,
                "labels": None,
                "models_geospatial": None,
                "models_metadata": None,
                "models_names": None,
                "prediction_scores": None,
                "task_types": None,
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }

    assert eval_job_bounded_area_1200_1800.results().metrics != expected_metrics
    assert (
        eval_job_bounded_area_1200_1800.results().metrics
        == eval_job_min_area_1200.results().metrics
    )


def test_get_bulk_evaluations(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    pred_dets2: list[Prediction],
):
    dataset_ = dataset_name
    model_ = model_name

    dataset = Dataset(client, dataset_)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model(client, model_)
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

    # test error when we don't pass either a model or dataset
    with pytest.raises(ValueError):
        client.get_bulk_evaluations()

    evaluations = client.get_bulk_evaluations(
        datasets=dataset_name, models=model_name
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].metrics)
    assert evaluations[0].metrics == expected_metrics

    # test incorrect names
    assert len(client.get_bulk_evaluations(datasets="wrong_dataset_name")) == 0
    assert len(client.get_bulk_evaluations(models="wrong_model_name")) == 0

    # test with multiple models
    second_model = Model(client, "second_model")
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
    assert second_model_evaluations[0].metrics == second_model_expected_metrics

    both_evaluations = client.get_bulk_evaluations(datasets=["test_dataset"])

    # should contain two different entries, one for each model
    assert len(both_evaluations) == 2
    assert all(
        [
            evaluation.model in ["second_model", model_name]
            for evaluation in both_evaluations
        ]
    )
    assert both_evaluations[0].metrics == expected_metrics
    assert both_evaluations[1].metrics == second_model_expected_metrics

    # should be equivalent since there are only two models attributed to this dataset
    both_evaluations_from_model_names = client.get_bulk_evaluations(
        models=["second_model", "test_model"]
    )
    assert len(both_evaluations_from_model_names) == 2
    assert both_evaluations[0] in both_evaluations_from_model_names
    assert both_evaluations[1] in both_evaluations_from_model_names

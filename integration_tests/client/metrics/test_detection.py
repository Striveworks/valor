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
from velour.schemas.filters import Filter
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

    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
    )
    assert isinstance(eval_job.evaluation_id, int)
    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []

    eval_results = eval_job.wait_for_completion()
    assert eval_results.status == JobStatus.DONE

    result = asdict(eval_results)
    assert result == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": TaskType.DETECTION,
        "settings": {
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "filters": {
                **default_filter_properties,
                "annotation_types": ["box"],
                "label_keys": ["k1"],
            },
        },
        "metrics": expected_metrics,
        "confusion_matrices": [],
        "status": JobStatus.DONE,
        "job_id": eval_job.evaluation_id,
    }

    # test evaluating a job using a value filter
    # these metrics should be the same as for eval_job
    # since all k1 keys had v1 values
    eval_job_value_filter_using_in_ = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.value.in_(["v1"]),
            Annotation.type == AnnotationType.BOX,
        ],
    )
    value_filter_result = asdict(
        eval_job_value_filter_using_in_.wait_for_completion(timeout=30)
    )
    assert value_filter_result["metrics"] == result["metrics"]

    # same as the above, but not using the in_ operator
    eval_job_value_filter = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.value == "v1",
            Annotation.type == AnnotationType.BOX,
        ],
    )
    value_filter_result = asdict(
        eval_job_value_filter.wait_for_completion(timeout=30)
    )
    assert value_filter_result["metrics"] == result["metrics"]

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
    )

    result = asdict(
        eval_job_bounded_area_10_2000.wait_for_completion(timeout=30)
    )
    assert result == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": TaskType.DETECTION,
        "settings": {
            "filters": {
                **default_filter_properties,
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
        },
        "metrics": expected_metrics,
        "confusion_matrices": [],
        "status": JobStatus.DONE,
        "job_id": eval_job_bounded_area_10_2000.evaluation_id,
    }

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
    )
    result = asdict(eval_job_min_area_1200.wait_for_completion(timeout=30))
    min_area_1200_metrics = result.pop("metrics")
    assert result == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": TaskType.DETECTION,
        "settings": {
            "filters": {
                **default_filter_properties,
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
        },
        # check metrics below
        "confusion_matrices": [],
        "status": JobStatus.DONE,
        "job_id": eval_job_min_area_1200.evaluation_id,
    }
    assert min_area_1200_metrics != expected_metrics

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
    )
    result = asdict(eval_job_max_area_1200.wait_for_completion(timeout=30))
    max_area_1200_metrics = result.pop("metrics")
    assert result == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": TaskType.DETECTION,
        "settings": {
            "filters": {
                **default_filter_properties,
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
        },
        # check metrics below
        "confusion_matrices": [],
        "status": JobStatus.DONE,
        "job_id": eval_job_max_area_1200.evaluation_id,
    }
    assert max_area_1200_metrics != expected_metrics

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
    )
    result = asdict(
        eval_job_bounded_area_1200_1800.wait_for_completion(timeout=30)
    )
    bounded_area_metrics = result.pop("metrics")
    assert result == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": TaskType.DETECTION,
        "settings": {
            "filters": {
                **default_filter_properties,
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
        },
        # check metrics below
        "confusion_matrices": [],
        "status": JobStatus.DONE,
        "job_id": eval_job_bounded_area_1200_1800.evaluation_id,
    }
    assert bounded_area_metrics != expected_metrics
    assert bounded_area_metrics == min_area_1200_metrics

    # test accessing these evaluations via the dataset
    all_evals = dataset.get_evaluations()
    assert len(all_evals) == 6


def test_evaluate_detection_with_json_filters(
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
):
    dataset = Dataset.create(client, dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
    for pd in pred_dets:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    # test default iou arguments
    eval_results = model.evaluate_detection(
        dataset=dataset,
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
    ).wait_for_completion(timeout=30)
    assert eval_results.settings.parameters.iou_thresholds_to_compute == [
        i / 100 for i in range(50, 100, 5)
    ]
    assert eval_results.settings.parameters.iou_thresholds_to_keep == [
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
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 1200,
        ],
    ).wait_for_completion(timeout=30)
    min_area_1200_metrics = eval_results_min_area_1200.metrics

    eval_job_bounded_area_1200_1800 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            **default_filter_properties,
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
    )

    result = asdict(
        eval_job_bounded_area_1200_1800.wait_for_completion(timeout=30)
    )
    bounded_area_metrics = result.pop("metrics")
    assert result == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": TaskType.DETECTION,
        "settings": {
            "filters": {
                **default_filter_properties,
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
        },
        # check metrics below
        "confusion_matrices": [],
        "status": JobStatus.DONE,
        "job_id": eval_job_bounded_area_1200_1800.evaluation_id,
    }
    assert bounded_area_metrics != expected_metrics
    assert bounded_area_metrics == min_area_1200_metrics


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
        client.get_bulk_evaluations()

    evaluations = client.get_bulk_evaluations(
        datasets=dataset_name, models=model_name
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].metrics)
    assert evaluations[0].metrics == expected_metrics

    evaluations_by_job_id = client.get_bulk_evaluations(
        job_ids=eval_job.job_id
    )
    assert len(evaluations_by_job_id) == 1
    assert evaluations_by_job_id[0] == evaluations[0]

    # test incorrect names
    assert len(client.get_bulk_evaluations(datasets="wrong_dataset_name")) == 0
    assert len(client.get_bulk_evaluations(models="wrong_model_name")) == 0

    # test with multiple models
    second_model = Model.create(client, "second_model")
    for pd in pred_dets2:
        second_model.add_prediction(pd)
    second_model.finalize_inferences(dataset)

    eval_job2 = second_model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
    )
    eval_job2.wait_for_completion(timeout=30)

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

    # should also be equivalent
    both_evaluations_from_job_ids = client.get_bulk_evaluations(
        job_ids=[eval_job.job_id, eval_job2.job_id]
    )
    assert len(both_evaluations_from_job_ids) == 2
    assert both_evaluations[0] in both_evaluations_from_job_ids
    assert both_evaluations[1] in both_evaluations_from_job_ids

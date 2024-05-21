""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import random
from dataclasses import asdict

import pytest
import requests
from geoalchemy2.functions import ST_Area
from sqlalchemy import select
from sqlalchemy.orm import Session

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    Filter,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import AnnotationType, EvaluationStatus, TaskType
from valor.exceptions import ClientException
from valor.schemas import Box
from valor_api.backend import models

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
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
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
    result_dict = result.to_dict()
    # duration isn't deterministic, so test meta separately
    assert result_dict["meta"]["datums"] == 2
    assert result_dict["meta"]["labels"] == 1  # we're filtering on one label
    assert result_dict["meta"]["annotations"] == 3
    assert result_dict["meta"]["duration"] <= 5
    result_dict.pop("meta")
    actual_metrics = result_dict.pop("metrics")

    assert result_dict == {
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
            "metrics": [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
        },
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test evaluating a job using a `Label.labels` filter
    eval_job_value_filter_using_in_ = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Annotation.labels == [Label(key="k1", value="v1")],
            Annotation.bounding_box.is_not_none(),
        ],
    )
    assert (
        eval_job_value_filter_using_in_.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )

    for m in eval_job_value_filter_using_in_.metrics:
        assert m in result.metrics
    for m in result.metrics:
        assert m in eval_job_value_filter_using_in_.metrics

    # same as the above, but not using the in_ operator
    eval_job_value_filter = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        filter_by=[
            Annotation.labels == [Label(key="k1", value="v1")],
        ],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert (
        eval_job_value_filter.wait_for_completion(timeout=30)
        == EvaluationStatus.DONE
    )

    for m in eval_job_value_filter.metrics:
        assert m in result.metrics
    for m in result.metrics:
        assert m in eval_job_value_filter.metrics

    # assert that this evaluation returns no metrics as there aren't any
    # Labels with key=k1 and value=v2
    with pytest.raises(ClientException) as e:
        model.evaluate_detection(
            dataset,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_return=[0.1, 0.6],
            filter_by=[
                Annotation.labels == [Label(key="k1", value="v2")],
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
    eval_job_bounded_area_10_2000_dict = (
        eval_job_bounded_area_10_2000.to_dict()
    )
    eval_job_bounded_area_10_2000_dict.pop("meta")
    actual_metrics = eval_job_bounded_area_10_2000_dict.pop("metrics")
    assert eval_job_bounded_area_10_2000_dict == {
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
            "metrics": [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
        },
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

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
    result.pop("meta")
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
            "metrics": [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
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
    # this computation will return 'EvaluationStatus.DONE' immediately as no predictions exist that meet the filter requirements.
    eval_job_max_area_1200.wait_for_completion(timeout=30)
    result = eval_job_max_area_1200.to_dict()
    result.pop("meta")
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
            "metrics": [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
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
    result.pop("meta")
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
            "metrics": [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
        },
        # check metrics below
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }
    assert bounded_area_metrics != expected_metrics
    for m in bounded_area_metrics:
        assert m in min_area_1200_metrics
    for m in min_area_1200_metrics:
        assert m in bounded_area_metrics

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
            Annotation.bounding_box.is_not_none(),
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
    result.pop("meta")
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
            "metrics": [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
        },
        # check metrics below
        "status": EvaluationStatus.DONE.value,
        "confusion_matrices": [],
        "missing_pred_labels": [],
        "ignored_pred_labels": [],
    }
    assert bounded_area_metrics != expected_metrics
    for m in bounded_area_metrics:
        assert m in min_area_1200_metrics
    for m in min_area_1200_metrics:
        assert m in bounded_area_metrics


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
            Annotation.bounding_box.is_not_none(),
        ],
    )
    eval_job.wait_for_completion(timeout=30)

    expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
    ]

    second_model_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.0,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.0,
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
    ]

    # test error when we don't pass either a model or dataset
    with pytest.raises(ValueError):
        client.get_evaluations()

    evaluations = client.get_evaluations(
        datasets=dataset_name, models=model_name  # type: ignore - purposefully throwing errors
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].metrics)
    for m in evaluations[0].metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in evaluations[0].metrics

    evaluations_by_evaluation_id = client.get_evaluations(
        evaluation_ids=eval_job.id  # type: ignore - purposefully throwing an error
    )
    assert len(evaluations_by_evaluation_id) == 1
    assert (
        evaluations_by_evaluation_id[0].to_dict() == evaluations[0].to_dict()
    )

    # test incorrect names
    assert len(client.get_evaluations(datasets="wrong_dataset_name")) == 0  # type: ignore - purposefully throwing an error
    assert len(client.get_evaluations(models="wrong_model_name")) == 0  # type: ignore - purposefully throwing an error

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
            Annotation.bounding_box.is_not_none(),
        ],
    )
    eval_job2.wait_for_completion(timeout=30)

    second_model_evaluations = client.get_evaluations(models=["second_model"])

    assert len(second_model_evaluations) == 1
    for m in second_model_evaluations[0].metrics:
        assert m in second_model_expected_metrics
    for m in second_model_expected_metrics:
        assert m in second_model_evaluations[0].metrics

    both_evaluations = client.get_evaluations(datasets=["test_dataset"])

    # should contain two different entries, one for each model
    assert len(both_evaluations) == 2
    for evaluation in both_evaluations:
        assert evaluation.model_name in [
            "second_model",
            model_name,
        ]
        if evaluation.model_name == model_name:
            for m in evaluation.metrics:
                assert m in expected_metrics
            for m in expected_metrics:
                assert m in evaluation.metrics
        elif evaluation.model_name == "second_model":
            for m in evaluation.metrics:
                assert m in second_model_expected_metrics
            for m in second_model_expected_metrics:
                assert m in evaluation.metrics

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

    # check that the content-range header exists on the raw response
    requests_method = getattr(requests, "get")
    resp = requests_method(
        "http://localhost:8000/evaluations?offset=1&limit=50"
    )
    assert resp.headers["content-range"] == "items 1-1/2"

    # test metrics_to_sort_by
    both_evaluations_from_evaluation_ids_sorted = client.get_evaluations(
        evaluation_ids=[eval_job.id, eval_job2.id],
        metrics_to_sort_by={"mAPAveragedOverIOUs": "k1"},
    )

    assert both_evaluations_from_evaluation_ids[0].metrics[-1]["value"] == 0

    # with sorting, the evaluation with the higher mAPAveragedOverIOUs is returned first
    assert (
        both_evaluations_from_evaluation_ids_sorted[0].metrics[-1]["value"]
        == 0.504950495049505
    )

    # test bad metrics_to_sort_by list
    with pytest.raises(ClientException):
        both_evaluations_from_evaluation_ids_sorted = client.get_evaluations(
            evaluation_ids=[eval_job.id, eval_job2.id],
            metrics_to_sort_by=["AP"],
        )


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
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
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
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
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
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
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
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
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
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
    )

    assert (
        eval_job.ignored_pred_labels is not None
        and eval_job.missing_pred_labels is not None
    )
    assert (
        len(eval_job.ignored_pred_labels) == 2
    )  # we're ignoring the two "cat" model predictions
    assert (
        len(eval_job.missing_pred_labels) == 3
    )  # we're missing three gts_det_syn representing different breeds of cats

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    pr_metrics = []
    for m in metrics:
        if m["type"] != "PrecisionRecallCurve":
            assert m in baseline_expected_metrics
        else:
            pr_metrics.append(m)

    for m in baseline_expected_metrics:
        assert m in metrics

    # TODO
    # pr_metrics.sort(key=lambda x: x["parameters"]["label_key"])

    # pr_expected_answers = {
    #     # class
    #     (
    #         0,
    #         "class",
    #         "cat",
    #         "0.1",
    #         "fp",
    #     ): 1,
    #     (0, "class", "cat", "0.4", "fp"): 0,
    #     (0, "class", "siamese cat", "0.1", "fn"): 1,
    #     (0, "class", "british shorthair", "0.1", "fn"): 1,
    #     # class_name
    #     (1, "class_name", "cat", "0.1", "fp"): 1,
    #     (1, "class_name", "maine coon cat", "0.1", "fn"): 1,
    #     # k1
    #     (2, "k1", "v1", "0.1", "fn"): 1,
    #     (2, "k1", "v1", "0.1", "tp"): 1,
    #     (2, "k1", "v1", "0.4", "fn"): 2,
    #     # k2
    #     (3, "k2", "v2", "0.1", "fn"): 1,
    #     (3, "k2", "v2", "0.1", "fp"): 1,
    # }

    # for (
    #     index,
    #     key,
    #     value,
    #     threshold,
    #     metric,
    # ), expected_length in pr_expected_answers.items():
    #     assert (
    #         len(pr_metrics[index]["value"][value][threshold][metric])
    #         == expected_length
    #     )

    # # spot check a few geojson results
    # assert (
    #     pr_metrics[0]["value"]["cat"]["0.1"]["fp"][0][2]
    #     == '{"type":"Polygon","coordinates":[[[10,10],[60,10],[60,40],[10,40],[10,10]]]}'
    # )
    # assert (
    #     pr_metrics[1]["value"]["maine coon cat"]["0.1"]["fn"][0][2]
    #     == '{"type":"Polygon","coordinates":[[[10,10],[60,10],[60,40],[10,40],[10,10]]]}'
    # )
    # assert (
    #     pr_metrics[3]["value"]["v2"]["0.1"]["fp"][0][2]
    #     == '{"type":"Polygon","coordinates":[[[15,0],[70,0],[70,20],[15,20],[15,0]]]}'
    # )

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
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.3333333333333333,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.3333333333333333,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": -1.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
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
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
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
    assert eval_job.ignored_pred_labels is not None
    assert eval_job.missing_pred_labels is not None

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

    # next, we check that the label mapping works when the label is completely foreign
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
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6666666666666666,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
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
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
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
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6666666666666666,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
    ]

    label_mapping = {
        # map the ground truths
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
    assert (
        eval_job.ignored_pred_labels is not None
        and eval_job.missing_pred_labels is not None
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

    # finally, let's test using a higher recall_score_threshold
    # this new threshold will disqualify all of our predictions for img1

    foo_expected_metrics_with_higher_score_threshold = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
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
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.3333333333333333,  # two missed groundtruth on the first image, and 1 hit for the second image
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
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
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.3333333333333333,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.0,
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
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        label_map=label_mapping,
        recall_score_threshold=0.8,
    )
    assert (
        eval_job.ignored_pred_labels is not None
        and eval_job.missing_pred_labels is not None
    )
    assert len(eval_job.ignored_pred_labels) == 0
    assert len(eval_job.missing_pred_labels) == 0
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    assert eval_job.to_dict()["parameters"] == {
        "task_type": "object-detection",
        "convert_annotations_to_type": None,
        "iou_thresholds_to_compute": [0.1, 0.6],
        "iou_thresholds_to_return": [0.1, 0.6],
        "label_map": [
            [["class_name", "maine coon cat"], ["foo", "bar"]],
            [["class", "siamese cat"], ["foo", "bar"]],
            [["class", "british shorthair"], ["foo", "bar"]],
            [["class", "cat"], ["foo", "bar"]],
            [["class_name", "cat"], ["foo", "bar"]],
        ],
        "metrics": [
            "AP",
            "AR",
            "mAP",
            "APAveragedOverIOUs",
            "mAR",
            "mAPAveragedOverIOUs",
        ],
        "recall_score_threshold": 0.8,
        "pr_curve_iou_threshold": 0.5,
        "pr_curve_max_examples": 1,
    }

    metrics = eval_job.metrics

    pr_metrics = []
    for m in metrics:
        if m["type"] != "PrecisionRecallCurve":
            assert m in foo_expected_metrics_with_higher_score_threshold
        else:
            pr_metrics.append(m)

    for m in foo_expected_metrics_with_higher_score_threshold:
        assert m in metrics

    # pr_metrics.sort(key=lambda x: x["parameters"]["label_key"])

    # TODO
    # pr_expected_answers = {
    #     # foo
    #     (0, "foo", "bar", "0.1", "fn"): 1,  # missed rect3
    #     (0, "foo", "bar", "0.1", "tp"): 2,
    #     (0, "foo", "bar", "0.4", "fn"): 2,
    #     (0, "foo", "bar", "0.4", "tp"): 1,
    #     # k1
    #     (1, "k1", "v1", "0.1", "fn"): 1,
    #     (1, "k1", "v1", "0.1", "tp"): 1,
    #     (1, "k1", "v1", "0.4", "fn"): 2,
    #     # k2
    #     (2, "k2", "v2", "0.1", "fn"): 1,
    #     (2, "k2", "v2", "0.1", "fp"): 1,
    # }

    # for (
    #     index,
    #     _,
    #     value,
    #     threshold,
    #     metric,
    # ), expected_length in pr_expected_answers.items():
    #     assert (
    #         len(pr_metrics[index]["value"][value][threshold][metric])
    #         == expected_length
    #     )

    # spot check a few geojson results
    # pr_metric = [
    #     m for m in pr_metrics if m["parameters"]["label_key"] == "foo"
    # ][0]
    # assert (
    #     pr_metric["value"]["bar"]["0.4"]["fn"][0][2]
    #     == '{"type":"Polygon","coordinates":[[[10,10],[60,10],[60,40],[10,40],[10,10]]]}'
    # )
    # assert (
    #     pr_metric["value"]["bar"]["0.4"]["tp"][0][2]
    #     == '{"type":"Polygon","coordinates":[[[15,0],[70,0],[70,20],[15,20],[15,0]]]}'
    # )

    # pr_metric = [
    #     m for m in pr_metrics if m["parameters"]["label_key"] == "k2"
    # ][0]
    # assert (
    #     pr_metric["value"]["v2"]["0.1"]["fp"][0][2]
    #     == '{"type":"Polygon","coordinates":[[[15,0],[70,0],[70,20],[15,20],[15,0]]]}'
    # )

    assert eval_job.parameters.label_map == [
        [["class_name", "maine coon cat"], ["foo", "bar"]],
        [["class", "siamese cat"], ["foo", "bar"]],
        [["class", "british shorthair"], ["foo", "bar"]],
        [["class", "cat"], ["foo", "bar"]],
        [["class_name", "cat"], ["foo", "bar"]],
    ]

    # test getting a subset of metrics
    choices = random.sample(
        [
            "AP",
            "AR",
            "mAP",
            "APAveragedOverIOUs",
            "mAR",
            "mAPAveragedOverIOUs",
        ],
        2,
    )

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_return=[0.1, 0.6],
        metrics=choices,
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metric_types = [m["type"] for m in eval_job.metrics]
    assert set(metric_types) == set(choices)


def test_evaluate_detection_false_negatives_single_image_baseline(
    db: Session, dataset_name: str, model_name: str, client: Client
):
    """This is the baseline for the below test. In this case there are two predictions and
    one groundtruth, but the highest confident prediction overlaps sufficiently with the groundtruth
    so there is not a penalty for the false negative so the AP is 1
    """
    dset = Dataset.create(dataset_name)
    dset.add_groundtruth(
        GroundTruth(
            datum=Datum(uid="uid1"),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    bounding_box=Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[Label(key="key", value="value")],
                )
            ],
        )
    )
    dset.finalize()

    model = Model.create(model_name)
    model.add_prediction(
        dset,
        Prediction(
            datum=Datum(uid="uid1"),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    bounding_box=Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[Label(key="key", value="value", score=0.8)],
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    bounding_box=Box.from_extrema(
                        xmin=100, xmax=110, ymin=100, ymax=200
                    ),
                    labels=[Label(key="key", value="value", score=0.7)],
                ),
            ],
        ),
    )

    evaluation = model.evaluate_detection(
        dset, iou_thresholds_to_compute=[0.5], iou_thresholds_to_return=[0.5]
    )
    evaluation.wait_for_completion(timeout=30)
    ap_metric = [m for m in evaluation.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 1,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_single_image(
    db: Session, dataset_name: str, model_name: str, client: Client
):
    """Tests fix for a bug where high confidence false negative was not being penalized. The
    difference between this test and the above is that here the prediction with higher confidence
    does not sufficiently overlap the groundtruth and so is penalized and we get an AP of 0.5
    """
    dset = Dataset.create(dataset_name)
    dset.add_groundtruth(
        GroundTruth(
            datum=Datum(uid="uid1"),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    bounding_box=Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[Label(key="key", value="value")],
                )
            ],
        )
    )
    dset.finalize()

    model = Model.create(model_name)
    model.add_prediction(
        dset,
        Prediction(
            datum=Datum(uid="uid1"),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    bounding_box=Box.from_extrema(
                        xmin=10, xmax=20, ymin=10, ymax=20
                    ),
                    labels=[Label(key="key", value="value", score=0.8)],
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    bounding_box=Box.from_extrema(
                        xmin=100, xmax=110, ymin=100, ymax=200
                    ),
                    labels=[Label(key="key", value="value", score=0.9)],
                ),
            ],
        ),
    )

    evaluation = model.evaluate_detection(
        dset, iou_thresholds_to_compute=[0.5], iou_thresholds_to_return=[0.5]
    )
    evaluation.wait_for_completion(timeout=30)

    ap_metric = [m for m in evaluation.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0.5,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_empty_low_confidence_of_fp(
    db: Session, dataset_name: str, model_name: str, client: Client
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation but a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP should be 1.0 since the false positive has lower confidence than the true positive

    """
    dset = Dataset.create(dataset_name)
    dset.add_groundtruths(
        [
            GroundTruth(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value")],
                    )
                ],
            ),
            GroundTruth(
                datum=Datum(uid="uid2"),
                annotations=[Annotation(task_type=TaskType.EMPTY)],
            ),
        ]
    )
    dset.finalize()

    model = Model.create(model_name)
    model.add_predictions(
        dset,
        [
            Prediction(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.8)],
                    ),
                ],
            ),
            Prediction(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.7)],
                    ),
                ],
            ),
        ],
    )

    evaluation = model.evaluate_detection(
        dset, iou_thresholds_to_compute=[0.5], iou_thresholds_to_return=[0.5]
    )
    evaluation.wait_for_completion(timeout=30)
    ap_metric = [m for m in evaluation.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 1.0,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_empty_high_confidence_of_fp(
    db: Session, dataset_name: str, model_name: str, client: Client
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class and high IOU)
        2. A second image with empty groundtruth annotation and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP should be 0.5 since the false positive has higher confidence than the true positive
    """
    dset = Dataset.create(dataset_name)
    dset.add_groundtruths(
        [
            GroundTruth(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value")],
                    )
                ],
            ),
            GroundTruth(
                datum=Datum(uid="uid2"),
                annotations=[Annotation(task_type=TaskType.EMPTY)],
            ),
        ]
    )
    dset.finalize()

    model = Model.create(model_name)
    model.add_predictions(
        dset,
        [
            Prediction(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.8)],
                    ),
                ],
            ),
            Prediction(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.9)],
                    ),
                ],
            ),
        ],
    )

    evaluation = model.evaluate_detection(
        dset, iou_thresholds_to_compute=[0.5], iou_thresholds_to_return=[0.5]
    )
    evaluation.wait_for_completion(timeout=30)
    ap_metric = [m for m in evaluation.metrics if m["type"] == "AP"][0]
    assert ap_metric == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0.5,
        "label": {"key": "key", "value": "value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_low_confidence_of_fp(
    db: Session, dataset_name: str, model_name: str, client: Client
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with class `"other value"` and a prediction with lower confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 1 since the false positive has lower confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    dset = Dataset.create(dataset_name)
    dset.add_groundtruths(
        [
            GroundTruth(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value")],
                    )
                ],
            ),
            GroundTruth(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="other value")],
                    )
                ],
            ),
        ]
    )
    dset.finalize()

    model = Model.create(model_name)
    model.add_predictions(
        dset,
        [
            Prediction(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.8)],
                    ),
                ],
            ),
            Prediction(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.7)],
                    ),
                ],
            ),
        ],
    )

    evaluation = model.evaluate_detection(
        dset, iou_thresholds_to_compute=[0.5], iou_thresholds_to_return=[0.5]
    )
    evaluation.wait_for_completion(timeout=30)
    ap_metric1 = [
        m
        for m in evaluation.metrics
        if m["type"] == "AP" and m["label"] == {"key": "key", "value": "value"}
    ][0]
    assert ap_metric1 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 1.0,
        "label": {"key": "key", "value": "value"},
    }

    # label `"other value"` is not in the predictions so we should get an AP of 0
    ap_metric2 = [
        m
        for m in evaluation.metrics
        if m["type"] == "AP"
        and m["label"] == {"key": "key", "value": "other value"}
    ][0]
    assert ap_metric2 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0,
        "label": {"key": "key", "value": "other value"},
    }


def test_evaluate_detection_false_negatives_two_images_one_only_with_different_class_high_confidence_of_fp(
    db: Session, dataset_name: str, model_name: str, client: Client
):
    """In this test we have
        1. An image with a matching groundtruth and prediction (same class, `"value"`, and high IOU)
        2. A second image with a groundtruth annotation with clas `"other value"` and a prediction with higher confidence
        then the prediction on the first image.

    In this case, the AP for class `"value"` should be 0.5 since the false positive has higher confidence than the true positive.
    AP for class `"other value"` should be 0 since there is no prediction for the `"other value"` groundtruth
    """
    dset = Dataset.create(dataset_name)
    dset.add_groundtruths(
        [
            GroundTruth(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value")],
                    )
                ],
            ),
            GroundTruth(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="other value")],
                    )
                ],
            ),
        ]
    )
    dset.finalize()

    model = Model.create(model_name)
    model.add_predictions(
        dset,
        [
            Prediction(
                datum=Datum(uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.8)],
                    ),
                ],
            ),
            Prediction(
                datum=Datum(uid="uid2"),
                annotations=[
                    Annotation(
                        task_type=TaskType.OBJECT_DETECTION,
                        bounding_box=Box.from_extrema(
                            xmin=10, xmax=20, ymin=10, ymax=20
                        ),
                        labels=[Label(key="key", value="value", score=0.9)],
                    ),
                ],
            ),
        ],
    )

    evaluation = model.evaluate_detection(
        dset, iou_thresholds_to_compute=[0.5], iou_thresholds_to_return=[0.5]
    )
    evaluation.wait_for_completion(timeout=30)
    ap_metric1 = [
        m
        for m in evaluation.metrics
        if m["type"] == "AP" and m["label"] == {"key": "key", "value": "value"}
    ][0]
    assert ap_metric1 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0.5,
        "label": {"key": "key", "value": "value"},
    }

    # label `"other value"` is not in the predictions so we should get an AP of 0
    ap_metric2 = [
        m
        for m in evaluation.metrics
        if m["type"] == "AP"
        and m["label"] == {"key": "key", "value": "other value"}
    ][0]
    assert ap_metric2 == {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0,
        "label": {"key": "key", "value": "other value"},
    }

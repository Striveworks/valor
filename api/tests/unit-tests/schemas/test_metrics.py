import pytest
from pydantic import ValidationError

from velour_api import enums, schemas


def test_metrics_DetectionParameters():
    schemas.DetectionParameters()

    schemas.DetectionParameters(
        iou_thresholds_to_compute=[0.2, 0.6],
        iou_thresholds_to_keep=[],
    )

    schemas.DetectionParameters(
        iou_thresholds_to_compute=[],
        iou_thresholds_to_keep=[],
    )

    schemas.DetectionParameters(
        iou_thresholds_to_compute=None,
        iou_thresholds_to_keep=[0.2],
    )

    with pytest.raises(ValidationError):
        schemas.DetectionParameters(
            iou_thresholds_to_compute=None,
            iou_thresholds_to_keep=0.2,
        )

    with pytest.raises(ValidationError):
        schemas.DetectionParameters(
            iou_thresholds_to_compute=[0.2, "test"],
            iou_thresholds_to_keep=[],
        )


def test_metrics_EvaluationSettings():
    schemas.EvaluationSettings()

    schemas.EvaluationSettings(
        parameters=schemas.DetectionParameters(
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_keep=[],
        ),
    )

    schemas.EvaluationSettings(
        parameters=schemas.DetectionParameters(
            iou_thresholds_to_compute=[],
            iou_thresholds_to_keep=[],
        ),
    )

    schemas.EvaluationSettings(
        parameters=schemas.DetectionParameters(
            iou_thresholds_to_compute=[],
            iou_thresholds_to_keep=[],
        ),
        filters=schemas.Filter(
            annotation_types=[enums.AnnotationType.BOX],
            label_keys=["class"],
        ),
    )

    with pytest.raises(ValidationError):
        schemas.EvaluationSettings(parameters="random_string")

    with pytest.raises(ValidationError):
        schemas.EvaluationSettings(filters="random_string")


def test_metrics_EvaluationJob():
    schemas.EvaluationJob(
        model="test_model",
        dataset="test_dataset",
        task_type=enums.TaskType.DETECTION,
        settings=schemas.EvaluationSettings(
            parameters=schemas.DetectionParameters(
                iou_thresholds_to_compute=[0.2, 0.6],
                iou_thresholds_to_keep=[0.2],
            ),
            filters=schemas.Filter(
                annotation_types=[enums.AnnotationType.BOX],
                label_keys=["k1"],
            ),
        ),
    )

    # test model argument errors
    with pytest.raises(ValidationError):
        schemas.EvaluationJob(
            model=123,
            dataset="test_dataset",
            task_type=enums.TaskType.DETECTION,
            settings=schemas.EvaluationSettings(
                parameters=schemas.DetectionParameters(
                    iou_thresholds_to_compute=[0.2, 0.6],
                    iou_thresholds_to_keep=[0.2],
                ),
                filters=schemas.Filter(
                    annotation_types=[enums.AnnotationType.BOX],
                    label_keys=["k1"],
                ),
            ),
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationJob(
            model=None,
            dataset="test_dataset",
            task_type=enums.TaskType.DETECTION,
            settings=schemas.EvaluationSettings(
                parameters=schemas.DetectionParameters(
                    iou_thresholds_to_compute=[0.2, 0.6],
                    iou_thresholds_to_keep=[0.2],
                ),
                filters=schemas.Filter(
                    annotation_types=[enums.AnnotationType.BOX],
                    label_keys=["k1"],
                ),
            ),
        )

        # test dataset errors
        with pytest.raises(ValidationError):
            schemas.EvaluationJob(
                model="model",
                dataset=None,
                task_type=enums.TaskType.DETECTION,
                settings=schemas.EvaluationSettings(
                    parameters=schemas.DetectionParameters(
                        iou_thresholds_to_compute=[0.2, 0.6],
                        iou_thresholds_to_keep=[0.2],
                    ),
                    filters=schemas.Filter(
                        annotation_types=[enums.AnnotationType.BOX],
                        label_keys=["k1"],
                    ),
                ),
            )

        with pytest.raises(ValidationError):
            schemas.EvaluationJob(
                model="model",
                dataset=123,
                task_type=enums.TaskType.DETECTION,
                settings=schemas.EvaluationSettings(
                    parameters=schemas.DetectionParameters(
                        iou_thresholds_to_compute=[0.2, 0.6],
                        iou_thresholds_to_keep=[0.2],
                    ),
                    filters=schemas.Filter(
                        annotation_types=[enums.AnnotationType.BOX],
                        label_keys=["k1"],
                    ),
                ),
            )

        # test task type and settings errors
        with pytest.raises(ValidationError):
            schemas.EvaluationJob(
                model="model",
                dataset="dataset",
                task_type="not a task type",
                settings=schemas.EvaluationSettings(
                    parameters=schemas.DetectionParameters(
                        iou_thresholds_to_compute=[0.2, 0.6],
                        iou_thresholds_to_keep=[0.2],
                    ),
                    filters=schemas.Filter(
                        annotation_types=[enums.AnnotationType.BOX],
                        label_keys=["k1"],
                    ),
                ),
            )

        with pytest.raises(ValidationError):
            schemas.EvaluationJob(
                model="model",
                dataset="dataset",
                task_type=None,
                settings=schemas.EvaluationSettings(
                    parameters=schemas.DetectionParameters(
                        iou_thresholds_to_compute=[0.2, 0.6],
                        iou_thresholds_to_keep=[0.2],
                    ),
                    filters=schemas.Filter(
                        annotation_types=[enums.AnnotationType.BOX],
                        label_keys=["k1"],
                    ),
                ),
            )

        with pytest.raises(ValidationError):
            schemas.EvaluationJob(
                model="model",
                dataset="dataset",
                task_type=enums.TaskType.DETECTION,
                settings=123,
            )

        with pytest.raises(ValidationError):
            schemas.EvaluationJob(
                model="model",
                dataset="dataset",
                task_type=enums.TaskType.DETECTION,
                settings=None,
            )


def test_metrics_CreateDetectionMetricsResponse():
    schemas.CreateDetectionMetricsResponse(
        missing_pred_labels=[], ignored_pred_labels=[], job_id=1
    )

    schemas.CreateDetectionMetricsResponse(
        missing_pred_labels=[schemas.Label(key="k1", value="v1")],
        ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
        job_id=123,
    )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=None,
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=schemas.Label(key="k1", value="v1"),
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=None,
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            job_id="not a job id",
        )


def test_metrics_CreateSemanticSegmentationMetricsResponse():
    schemas.CreateDetectionMetricsResponse(
        missing_pred_labels=[], ignored_pred_labels=[], job_id=1
    )

    schemas.CreateSemanticSegmentationMetricsResponse(
        missing_pred_labels=[schemas.Label(key="k1", value="v1")],
        ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
        job_id=123,
    )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=None,
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=schemas.Label(key="k1", value="v1"),
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=None,
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            job_id="not a job id",
        )


def test_metrics_CreateClfMetricsResponse():
    schemas.CreateClfMetricsResponse(
        missing_pred_keys=["k1", "k2"],
        ignored_pred_keys=["k1", "k2"],
        job_id=123,
    )

    schemas.CreateClfMetricsResponse(
        missing_pred_keys=[],
        ignored_pred_keys=[],
        job_id=123,
    )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys=None,
            ignored_pred_keys=["k1", "k2"],
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys="k1",
            ignored_pred_keys=["k1", "k2"],
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys=["k1", "k2"],
            ignored_pred_keys=None,
            job_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys=["k1", "k2"],
            ignored_pred_keys=["k1", "k2"],
            job_id="not a job id",
        )


def test_metrics_Job():
    schemas.metrics.Job(
        uid="uid",
        status=enums.JobStatus.PENDING,
    )

    with pytest.raises(ValidationError):
        schemas.metrics.Job(
            uid=123,
            status=enums.JobStatus.PENDING,
        )

    with pytest.raises(ValidationError):
        schemas.metrics.Job(
            uid="uid",
            status="not a status",
        )


def test_metrics_Metric():
    schemas.Metric(
        type="detection",
        parameters={},
        value=0.2,
        label=schemas.Label(key="k1", value="v1"),
    )

    schemas.Metric(type="detection")

    with pytest.raises(ValidationError):
        schemas.Metric(
            type="detection",
            parameters=123,
            value=0.2,
            label=schemas.Label(key="k1", value="v1"),
        )


# @TODO
def test_metrics_APMetric():
    pass


# @TODO
def test_metrics_APMetricAveragedOverIOUs():
    pass


# @TODO
def test_metrics_mAPMetric():
    pass


# @TODO
def test_metrics_mAPMetricAveragedOverIOUs():
    pass


# @TODO
def test_metrics_ConfusionMatrixEntry():
    pass


# @TODO
def test_metrics__BaseConfusionMatrix():
    pass


# @TODO
def test_metrics_ConfusionMatrix():
    pass


# @TODO
def test_metrics_ConfusionMatrixResponse():
    pass


# @TODO
def test_metrics_AccuracyMetric():
    pass


# @TODO
def test_metrics__PrecisionRecallF1Base():
    pass


# @TODO
def test_metrics_PrecisionMetric():
    pass


# @TODO
def test_metrics_RecallMetric():
    pass


# @TODO
def test_metrics_F1Metric():
    pass


# @TODO
def test_metrics_ROCAUCMetric():
    pass

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

    with pytest.raises(ValidationError):
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
        missing_pred_labels=[], ignored_pred_labels=[], evaluation_id=1
    )

    schemas.CreateDetectionMetricsResponse(
        missing_pred_labels=[schemas.Label(key="k1", value="v1")],
        ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
        evaluation_id=123,
    )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=None,
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=schemas.Label(key="k1", value="v1"),
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=None,
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateDetectionMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            evaluation_id="not a job id",
        )


def test_metrics_CreateSemanticSegmentationMetricsResponse():
    schemas.CreateDetectionMetricsResponse(
        missing_pred_labels=[], ignored_pred_labels=[], evaluation_id=1
    )

    schemas.CreateSemanticSegmentationMetricsResponse(
        missing_pred_labels=[schemas.Label(key="k1", value="v1")],
        ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
        evaluation_id=123,
    )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=None,
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=schemas.Label(key="k1", value="v1"),
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=None,
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateSemanticSegmentationMetricsResponse(
            missing_pred_labels=[schemas.Label(key="k1", value="v1")],
            ignored_pred_labels=[schemas.Label(key="k2", value="v2")],
            evaluation_id="not a job id",
        )


def test_metrics_CreateClfMetricsResponse():
    schemas.CreateClfMetricsResponse(
        missing_pred_keys=["k1", "k2"],
        ignored_pred_keys=["k1", "k2"],
        evaluation_id=123,
    )

    schemas.CreateClfMetricsResponse(
        missing_pred_keys=[],
        ignored_pred_keys=[],
        evaluation_id=123,
    )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys=None,
            ignored_pred_keys=["k1", "k2"],
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys="k1",
            ignored_pred_keys=["k1", "k2"],
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys=["k1", "k2"],
            ignored_pred_keys=None,
            evaluation_id=123,
        )

    with pytest.raises(ValidationError):
        schemas.CreateClfMetricsResponse(
            missing_pred_keys=["k1", "k2"],
            ignored_pred_keys=["k1", "k2"],
            evaluation_id="not a job id",
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


def test_metrics_APMetric():
    ap_metric = schemas.APMetric(
        iou=0.2, value=0.5, label=schemas.Label(key="k1", value="v1")
    )

    with pytest.raises(ValidationError):
        schemas.APMetric(
            iou=None, value=0.5, label=schemas.Label(key="k1", value="v1")
        )

    with pytest.raises(ValidationError):
        schemas.APMetric(
            iou=0.1, value=None, label=schemas.Label(key="k1", value="v1")
        )

    with pytest.raises(ValidationError):
        schemas.APMetric(iou=0.1, value=0.5, label="k1")

    assert all(
        [
            key in ["value", "label_id", "type", "evaluation_id", "parameters"]
            for key in ap_metric.db_mapping(label_id=1, evaluation_id=1)
        ]
    )


def test_metrics_APMetricAveragedOverIOUs():
    ap_averaged_metric = schemas.APMetricAveragedOverIOUs(
        ious=set([0.1, 0.2]),
        value=0.5,
        label=schemas.Label(key="k1", value="v1"),
    )

    with pytest.raises(ValidationError):
        schemas.APMetricAveragedOverIOUs(
            ious=None, value=0.5, label=schemas.Label(key="k1", value="v1")
        )

    with pytest.raises(ValidationError):
        schemas.APMetricAveragedOverIOUs(
            ious=set([0.1, 0.2]),
            value=None,
            label=schemas.Label(key="k1", value="v1"),
        )

    with pytest.raises(ValidationError):
        schemas.APMetricAveragedOverIOUs(
            ious=set([0.1, 0.2]), value=0.5, label="k1"
        )

    assert all(
        [
            key in ["value", "label_id", "type", "evaluation_id", "parameters"]
            for key in ap_averaged_metric.db_mapping(
                label_id=1, evaluation_id=1
            )
        ]
    )


def test_metrics_mAPMetric():
    map_metric = schemas.mAPMetric(iou=0.2, value=0.5)

    with pytest.raises(ValidationError):
        schemas.mAPMetric(iou=None, value=0.5)

    with pytest.raises(ValidationError):
        schemas.mAPMetric(iou=0.1, value=None)

    with pytest.raises(ValidationError):
        schemas.mAPMetric(iou=0.1, value="value")

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in map_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_metrics_mAPMetricAveragedOverIOUs():
    map_averaged_metric = schemas.mAPMetricAveragedOverIOUs(
        ious=set([0.1, 0.2]), value=0.5
    )

    with pytest.raises(ValidationError):
        schemas.mAPMetricAveragedOverIOUs(ious=None, value=0.5)

    with pytest.raises(ValidationError):
        schemas.mAPMetricAveragedOverIOUs(ious=set([0.1, 0.2]), value=None)

    with pytest.raises(ValidationError):
        schemas.mAPMetricAveragedOverIOUs(ious=set([0.1, 0.2]), value="value")

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in map_averaged_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_metrics_ConfusionMatrixEntry():
    schemas.ConfusionMatrixEntry(
        prediction="pred", groundtruth="gt", count=123
    )

    with pytest.raises(ValidationError):
        schemas.ConfusionMatrixEntry(
            prediction=None, groundtruth="gt", count=123
        )

    with pytest.raises(ValidationError):
        schemas.ConfusionMatrixEntry(
            prediction="pred", groundtruth=123, count=123
        )

    with pytest.raises(ValidationError):
        schemas.ConfusionMatrixEntry(
            prediction="pred", groundtruth="gt", count="not an int"
        )


def test_metrics__BaseConfusionMatrix():
    schemas.metrics._BaseConfusionMatrix(
        label_key="label",
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction="pred1", groundtruth="gt1", count=123
            ),
            schemas.ConfusionMatrixEntry(
                prediction="pred2", groundtruth="gt2", count=234
            ),
        ],
    )

    with pytest.raises(ValidationError):
        schemas.metrics._BaseConfusionMatrix(
            label_key=123,
            entries=[
                schemas.ConfusionMatrixEntry(
                    prediction="pred1", groundtruth="gt1", count=123
                ),
                schemas.ConfusionMatrixEntry(
                    prediction="pred2", groundtruth="gt2", count=234
                ),
            ],
        )

    with pytest.raises(ValidationError):
        schemas.metrics._BaseConfusionMatrix(label_key="label", entries=None)

    with pytest.raises(ValidationError):
        schemas.metrics._BaseConfusionMatrix(
            label_key="label", entries=["not an entry"]
        )


def test_metrics_ConfusionMatrix():
    confusion_matrix = schemas.metrics.ConfusionMatrix(
        label_key="label",
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction="pred1", groundtruth="gt1", count=123
            ),
            schemas.ConfusionMatrixEntry(
                prediction="pred2", groundtruth="gt2", count=234
            ),
        ],
    )

    with pytest.raises(ValidationError):
        schemas.metrics.ConfusionMatrix(
            label_key=123,
            entries=[
                schemas.ConfusionMatrixEntry(
                    prediction="pred1", groundtruth="gt1", count=123
                ),
                schemas.ConfusionMatrixEntry(
                    prediction="pred2", groundtruth="gt2", count=234
                ),
            ],
        )

    with pytest.raises(ValidationError):
        schemas.metrics.ConfusionMatrix(label_key="label", entries=None)

    with pytest.raises(ValidationError):
        schemas.metrics.ConfusionMatrix(
            label_key="label", entries=["not an entry"]
        )

    assert all(
        [
            key in ["label_key", "value", "evaluation_id"]
            for key in confusion_matrix.db_mapping(evaluation_id=1)
        ]
    )


def test_metrics_AccuracyMetric():
    acc_metric = schemas.AccuracyMetric(label_key="key", value=0.5)

    with pytest.raises(ValidationError):
        schemas.AccuracyMetric(label_key=None, value=0.5)

    with pytest.raises(ValidationError):
        schemas.AccuracyMetric(label_key="key", value="value")

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in acc_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_metrics__PrecisionRecallF1Base():
    schemas.metrics._PrecisionRecallF1Base(
        label=schemas.Label(key="key", value="value"), value=0.5
    )

    null_value = schemas.metrics._PrecisionRecallF1Base(
        label=schemas.Label(key="key", value="value"), value=None
    )

    assert null_value.value == -1

    with pytest.raises(ValidationError):
        schemas.metrics._PrecisionRecallF1Base(label=None, value=0.5)

    with pytest.raises(ValidationError):
        schemas.metrics._PrecisionRecallF1Base(
            label=schemas.Label(key="key", value="value"), value="value"
        )


def test_metrics_PrecisionMetric():
    precision_recall_metric = schemas.metrics.PrecisionMetric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.db_mapping(label_id=1, evaluation_id=2)

    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in mapping
        ]
    )

    assert mapping["type"] == "Precision"


def test_metrics_RecallMetric():
    precision_recall_metric = schemas.metrics.RecallMetric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.db_mapping(label_id=1, evaluation_id=2)

    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in mapping
        ]
    )

    assert mapping["type"] == "Recall"


def test_metrics_F1Metric():
    precision_recall_metric = schemas.metrics.F1Metric(
        label=schemas.Label(key="key", value="value"), value=0.5
    )
    mapping = precision_recall_metric.db_mapping(label_id=1, evaluation_id=2)

    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in mapping
        ]
    )

    assert mapping["type"] == "F1"


def test_metrics_ROCAUCMetric():
    roc_auc_metric = schemas.ROCAUCMetric(label_key="key", value=0.2)

    with pytest.raises(ValidationError):
        schemas.ROCAUCMetric(label_key=None, value=0.2)

    with pytest.raises(ValidationError):
        schemas.ROCAUCMetric(label_key=123, value=0.2)

    with pytest.raises(ValidationError):
        schemas.ROCAUCMetric(label_key="key", value=None)

    with pytest.raises(ValidationError):
        schemas.ROCAUCMetric(label_key="key", value="not a number")

    assert all(
        [
            key in ["value", "type", "evaluation_id", "parameters"]
            for key in roc_auc_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_metrics_IOUMetric():
    iou_metric = schemas.IOUMetric(
        label=schemas.Label(key="key", value="value"), value=0.2
    )

    with pytest.raises(ValidationError):
        schemas.IOUMetric(label=None, value=0.2)

    with pytest.raises(ValidationError):
        schemas.IOUMetric(label="not a label", value=0.2)

    with pytest.raises(ValidationError):
        schemas.IOUMetric(
            label=schemas.Label(key="key", value="value"), value=None
        )

    with pytest.raises(ValidationError):
        schemas.IOUMetric(
            label=schemas.Label(key="key", value="value"), value="not a value"
        )
    assert all(
        [
            key in ["value", "type", "evaluation_id", "label_id"]
            for key in iou_metric.db_mapping(evaluation_id=1, label_id=2)
        ]
    )


def test_metrics_mIOUMetric():
    iou_metric = schemas.mIOUMetric(value=0.2)

    with pytest.raises(ValidationError):
        schemas.mIOUMetric(value=None)

    with pytest.raises(ValidationError):
        schemas.mIOUMetric(value="not a value")

    assert all(
        [
            key in ["value", "type", "evaluation_id"]
            for key in iou_metric.db_mapping(evaluation_id=1)
        ]
    )


def test_Evaluation():
    schemas.Evaluation(
        dataset="dataset",
        model="model",
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(),
        evaluation_id=1,
        status="done",
        metrics=[],
        confusion_matrices=[],
    )

    with pytest.raises(ValidationError):
        schemas.Evaluation(
            dataset=123,
            model="model",
            task_type=enums.TaskType.CLASSIFICATION,
            settings=schemas.EvaluationSettings(),
            evaluation_id=1,
            status="done",
            metrics=[],
            confusion_matrices=[],
        )

    with pytest.raises(ValidationError):
        schemas.Evaluation(
            dataset="dataset",
            model=None,
            task_type=enums.TaskType.CLASSIFICATION,
            settings=schemas.EvaluationSettings(),
            evaluation_id=1,
            status="done",
            metrics=[],
            confusion_matrices=[],
        )

    with pytest.raises(ValidationError):
        schemas.Evaluation(
            dataset="dataset",
            model="model",
            task_type=enums.TaskType.CLASSIFICATION,
            settings=123,
            evaluation_id=1,
            status="done",
            metrics=[],
            confusion_matrices=[],
        )

    with pytest.raises(ValidationError):
        schemas.Evaluation(
            dataset="dataset",
            model="model",
            task_type=enums.TaskType.CLASSIFICATION,
            settings=schemas.EvaluationSettings(),
            evaluation_id="not a job id",
            status="done",
            metrics=[],
            confusion_matrices=[],
        )

    with pytest.raises(ValidationError):
        schemas.Evaluation(
            dataset="dataset",
            model="model",
            task_type=enums.TaskType.CLASSIFICATION,
            settings=schemas.EvaluationSettings(),
            evaluation_id=1,
            status=123,
            metrics=[],
            confusion_matrices=[],
        )

    with pytest.raises(ValidationError):
        schemas.Evaluation(
            dataset="dataset",
            model="model",
            task_type=enums.TaskType.CLASSIFICATION,
            settings=schemas.EvaluationSettings(),
            evaluation_id=1,
            status="done",
            metrics=None,
            confusion_matrices=[],
        )

    with pytest.raises(ValidationError):
        schemas.Evaluation(
            dataset="dataset",
            model="model",
            task_type=enums.TaskType.CLASSIFICATION,
            settings=schemas.EvaluationSettings(),
            evaluation_id=1,
            status="done",
            metrics=[],
            confusion_matrices=None,
        )

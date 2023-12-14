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


# @TODO
def test_metrics_APRequest():
    pass


# @TODO
def test_metrics_CreateDetectionMetricsResponse():
    pass


# @TODO
def test_metrics_CreateClfMetricsResponse():
    pass


# @TODO
def test_metrics_Job():
    pass


# @TODO
def test_metrics_ClfMetricsRequest():
    pass


# @TODO
def test_metrics_Metric():
    pass


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

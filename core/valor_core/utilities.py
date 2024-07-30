import pandas as pd
from valor_core import schemas, enums


# TODO these shouldn't be private
def _validate_parameters(
    parameters: schemas.EvaluationParameters, task_type: enums.TaskType
) -> schemas.EvaluationParameters:
    if parameters.metrics_to_return is None:
        parameters.metrics_to_return = {
            enums.TaskType.CLASSIFICATION: [
                enums.MetricType.Precision,
                enums.MetricType.Recall,
                enums.MetricType.F1,
                enums.MetricType.Accuracy,
                enums.MetricType.ROCAUC,
                enums.MetricType.PrecisionRecallCurve,
            ],
            enums.TaskType.OBJECT_DETECTION: [
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
        }[task_type]

    return parameters


def _validate_groundtruth_dataframe(
    df: pd.DataFrame, task_type: enums.TaskType
):
    pass


def _validate_prediction_dataframe(
    df: pd.DataFrame, task_type: enums.TaskType
):
    pass

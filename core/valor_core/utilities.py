import pandas as pd
from valor_core import schemas


# TODO these shouldn't be private
def _validate_parameters(parameters: schemas.EvaluationParameters):
    if parameters.metrics_to_return is None:
        raise RuntimeError("Metrics to return should always be defined here.")


def _validate_groundtruth_dataframe(df: pd.DataFrame, task_type: str):
    pass


def _validate_prediction_dataframe(df: pd.DataFrame, task_type: str):
    pass

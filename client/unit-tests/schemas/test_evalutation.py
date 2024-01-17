from dataclasses import asdict

import pytest

try:
    import pandas as pd  # noqa: F401
except ModuleNotFoundError:
    pd = None

from velour import enums, schemas


def test_evaluation_evaluation_job():
    params = {
        "model_filter": {
            "model_names": ["md"],
        },
        "evaluation_filter": {
            "dataset_names": ["ds"],
            "task_types": [enums.TaskType.DETECTION.value],
            "annotation_types": [enums.AnnotationType.BOX.value],
        },
        "parameters": asdict(schemas.EvaluationParameters()),
    }
    schemas.EvaluationRequest(**params)


@pytest.mark.skipif(pd is None, reason="pandas package is not installed")
def test_to_dataframe():
    try:
        import pandas as pd  # noqa: F401
    except ModuleNotFoundError:
        return

    def _generate_metric(
        type: str,
        parameters: dict = None,
        value: float = None,
        label: dict = None,
    ):
        return dict(type=type, parameters=parameters, value=value, label=label)

    df = schemas.EvaluationResponse(
        id=1,
        model_filter=schemas.Filter(
            model_names=["model1"],
            dataset_names=["dataset1"],
        ),
        evaluation_filter=schemas.Filter(
            model_names=["model1"],
            dataset_names=["dataset1"],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
        parameters=schemas.EvaluationParameters(),
        status=enums.EvaluationStatus.DONE,
        metrics=[
            _generate_metric(
                "d",
                parameters={"x": 0.123, "y": 0.987},
                value=0.3,
                label={"key": "k1", "value": "v2"},
            ),
            _generate_metric("a", value=0.99),
            _generate_metric("b", value=0.3),
            _generate_metric(
                "c", parameters={"x": 0.123, "y": 0.987}, value=0.3
            ),
            _generate_metric(
                "d",
                parameters={"x": 0.123, "y": 0.987},
                value=0.3,
                label={"key": "k1", "value": "v1"},
            ),
        ],
        confusion_matrices=[],
    ).to_dataframe()

    df_str = """                                        value
    dataset                              dataset1
    type parameters               label
    a    "n/a"                    n/a        0.99
    b    "n/a"                    n/a        0.30
    c    {"x": 0.123, "y": 0.987} n/a        0.30
    d    {"x": 0.123, "y": 0.987} k1: v1     0.30
                                  k1: v2     0.30"""

    assert str(df).replace(" ", "") == df_str.replace(" ", "")

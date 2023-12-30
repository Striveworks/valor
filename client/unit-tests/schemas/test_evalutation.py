from velour import enums, schemas


def test_evaluation_evaluation_job():
    params = {
        "model": "md",
        "dataset": "ds",
        "task_type": "object-detection",
        "settings": {
            "filters": {
                "annotation_types": [enums.AnnotationType.BOX],
            }
        },
        "id": None,
    }
    schemas.EvaluationJob(**params)

    params["id"] = 123
    schemas.EvaluationJob(**params)


def test_to_dataframe():

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return

    def _generate_metric(
        type: str,
        parameters: dict = None,
        value: float = None,
        label: dict = None,
    ):
        return dict(type=type, parameters=parameters, value=value, label=label)

    df = schemas.EvaluationResult(
        dataset="dataset1",
        model="model1",
        task_type=enums.TaskType.CLASSIFICATION,
        settings=schemas.EvaluationSettings(),
        job_id=1,
        status=enums.JobStatus.DONE,
        metrics=[
            _generate_metric("d", parameters={"x": 0.123, "y": 0.987}, value=0.3, label={"key":"k1", "value":"v2"}),
            _generate_metric("a", value=0.99),
            _generate_metric("b", value=0.3),
            _generate_metric("c", parameters={"x": 0.123, "y": 0.987}, value=0.3),
            _generate_metric("d", parameters={"x": 0.123, "y": 0.987}, value=0.3, label={"key":"k1", "value":"v1"}),
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
    
    assert str(df) == df_str

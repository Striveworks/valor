from dataclasses import asdict

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

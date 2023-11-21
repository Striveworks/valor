from velour import enums, schemas


def test_evaluation_evaluation_job():
    params = {
        "model": "md",
        "dataset": "ds",
        "task_type": "object-detection",
        "settings": {
            "filters": {
                "annotations": [enums.AnnotationType.BOX],
            }
        },
        "id": None,
    }
    schemas.EvaluationJob(**params)

    params["id"] = 123
    schemas.EvaluationJob(**params)

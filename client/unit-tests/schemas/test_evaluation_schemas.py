from velour import enums, schemas


def test_evaluation_request():
    params = {
        "model_names": ["md"],
        "datum_filter": {
            "dataset_names": ["ds"],
        },
        "parameters": {
            "task_type": enums.TaskType.DETECTION.value,
            "convert_annotations_to_type": enums.AnnotationType.BOX.value,
        },
    }
    schemas.EvaluationRequest(**params)

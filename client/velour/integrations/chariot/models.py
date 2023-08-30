from typing import Tuple

from velour import enums
from velour.client import Client, ClientException, Model
from velour.schemas import (
    BoundingBox,
    Datum,
    Label,
    Prediction,
    ScoredAnnotation,
    ScoredLabel,
)

""" Parsers """


def _parse_chariot_predict_image_classification(
    datum: Datum,
    labels: dict,
    result: list,
    label_key: str = "class_label",
):
    # validate result
    if not isinstance(result, list):
        raise TypeError(
            "image classification expected format List[List[float]]"
        )
    if len(result) != 1:
        raise ValueError("cannot have more than one result per datum")
    if not isinstance(result[0], str):
        raise TypeError(
            "image classification expected format List[List[float]]"
        )

    # create prediction
    return Prediction(
        datum=datum,
        annotations=[
            ScoredAnnotation(
                task_type=enums.TaskType.CLASSIFICATION,
                scored_labels=[
                    ScoredLabel(
                        label=Label(key=label_key, value=label),
                        score=1.0 if label == result[0] else 0.0,
                    )
                    for label in labels
                ],
            )
        ],
    )


def _parse_chariot_predict_proba_image_classification(
    datum: Datum,
    labels: dict,
    result: list,
    label_key: str = "class_label",
):
    # validate result
    if not isinstance(result, list):
        raise TypeError("image classification expected format List[str]")
    if len(result) != 1:
        raise ValueError("cannot have more than one result per datum")
    if not isinstance(result[0], list):
        raise TypeError("image classification expected format List[str]")
    if len(labels) != len(result[0]):
        raise ValueError("number of labels does not equal number of scores")

    # create prediction
    labels = {v: k for k, v in labels.items()}
    return Prediction(
        datum=datum,
        annotations=[
            ScoredAnnotation(
                task_type=enums.TaskType.CLASSIFICATION,
                scored_labels=[
                    ScoredLabel(
                        label=Label(key=label_key, value=labels[i]),
                        score=score,
                    )
                    for i, score in enumerate(result[0])
                ],
            )
        ],
    )


def _parse_chariot_detect_image_object_detection(
    datum: Datum,
    result: dict,
    label_key: str = "class_label",
):
    # validate result
    if not isinstance(result, list):
        raise TypeError("image object detection expected format List[Dict[]]")
    if len(result) != 1:
        raise ValueError("cannot have more than one result per datum")
    if not isinstance(result[0], dict):
        raise TypeError("image object detection expected format List[Dict[]]")
    result = result[0]

    # validate result
    expected_keys = {
        "num_detections",
        "detection_classes",
        "detection_boxes",
        "detection_scores",
    }
    if set(result.keys()) != expected_keys:
        raise ValueError(
            f"Expected `dets` to have keys {expected_keys} but got {result.keys()}"
        )

    # create prediction
    return Prediction(
        datum=datum,
        annotations=[
            ScoredAnnotation(
                task_type=enums.TaskType.DETECTION,
                scored_labels=[
                    ScoredLabel(
                        label=Label(key=label_key, value=label),
                        score=float(score),
                    )
                ],
                bounding_box=BoundingBox.from_extrema(
                    ymin=box[0], xmin=box[1], ymax=box[2], xmax=box[3]
                ),
            )
            for box, score, label in zip(
                result["detection_boxes"],
                result["detection_scores"],
                result["detection_classes"],
            )
        ],
    )


""" User """


def create_model_from_chariot(
    client: Client,
    model,
):
    return Model.create(
        client=client,
        name=model.id,
        integration="chariot",
        title=model.name,
        description=model._meta.summary,
        project_id=model.project_id,
    )


def get_prediction_parser_from_chariot(
    task_type: str, action: str, label_key: str, class_labels: list = None
):

    if task_type == "Image Classification":

        if action == "predict":

            def velour_parser(datum: Datum, result):
                return _parse_chariot_predict_image_classification(
                    datum, class_labels, result=result, label_key=label_key
                )

        elif action == "predict_proba":

            def velour_parser(datum: Datum, result):
                return _parse_chariot_predict_proba_image_classification(
                    datum, class_labels, result=result, label_key=label_key
                )

        else:
            raise NotImplementedError(
                f"action `{action}` not supported for task type `Image Classification`"
            )

    elif task_type == "Object Detection":

        def velour_parser(datum: Datum, result):
            return _parse_chariot_detect_image_object_detection(
                datum=datum,
                result=result,
                label_key=label_key,
            )

    elif task_type == "Image Segmentation":
        raise NotImplementedError(
            "Image Segmentation has not been implemented."
        )
    else:
        raise NotImplementedError(task_type)

    return velour_parser


def get_chariot_model_integration(
    client: Client, model, action: str, label_key: str = "class_label"
) -> Tuple[Model, callable]:
    """Returns tuple of (velour.client.Model, parsing_fn(datum, result))"""

    # get or create model
    try:
        velour_model = Model.get(client, model.id)
    except ClientException as e:
        if "does not exist" not in str(e):
            raise e
        velour_model = create_model_from_chariot(client, model.id)

    # retrieve task-related parser
    velour_parser = get_prediction_parser_from_chariot(
        task_type=model.task.value,
        action=action,
        label_key=label_key,
        class_labels=model.class_labels,
    )

    return (velour_model, velour_parser)

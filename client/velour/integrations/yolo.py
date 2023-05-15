from pathlib import Path
from typing import List

from velour.data_types import (
    Image,
    Label,
    PredictedImageClassification,
    ScoredLabel,
)


def parse_yolo_image_classification(result):

    # Extract data
    image_uid = Path(result.path).stem
    image_height = result.orig_shape[0]
    image_width = result.orig_shape[1]
    probabilities = result.probs
    labels = result.names

    # Create scored label list
    scored_labels = [
        ScoredLabel(
            label=Label(key="class_label", value=labels[key]),
            score=probability.item(),
        )
        for key, probability in list(zip(labels, probabilities))
    ]

    return PredictedImageClassification(
        image=Image(
            uid=image_uid,
            height=image_height,
            width=image_width,
        ),
        scored_labels=scored_labels,
    )


def parse_yolo_image_segmentation(result):

    # Extract data
    # image_uid = Path(result.path).stem
    # image_height = result.orig_shape[0]
    # image_width = result.orig_shape[1]

    pass


def parse_yolo_object_detection(result):

    # Extract data
    # image_uid = Path(result.path).stem
    # image_height = result.orig_shape[0]
    # image_width = result.orig_shape[1]

    # for mem in dir(result.boxes):
    #     print(mem)
    # print(result.boxes)

    pass


def upload_yolo_inferences(results: List[dict]):
    for result in results:

        if "masks" in result.keys and "boxes" in result.keys:
            parse_yolo_image_segmentation(result=result)
        elif "boxes" in result.keys:
            parse_yolo_object_detection(result=result)
        elif "probs" in result.keys:
            parse_yolo_image_classification(result=result)

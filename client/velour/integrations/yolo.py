from pathlib import Path
from typing import List

from velour.data_types import (
    Image,
    Label,
    PredictedImageClassification,
    PredictedInstanceSegmentation,
    ScoredLabel,
)


def parse_image_classification(result):

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


def parse_image_segmentation(result):

    # Extract data
    image_uid = Path(result.path).stem
    image_height = result.orig_shape[0]
    image_width = result.orig_shape[1]
    labels = result.names
    masks = result.masks.xy
    # boxes = result.boxes.xyxy
    confidence = [conf.item() for conf in result.boxes.conf]
    prediction = [labels[int(pred.item())] for pred in result.boxes.cls]

    print(result)
    print(f"Labels Shape: {len(result.names.keys())}")
    print(f"Masks Shape: {type(result.masks.xy)}")
    print(f"Boxes Shape: {type(result.boxes.xyxy)}")

    # Create scored label list
    scored_labels = [
        ScoredLabel(
            label=Label(key="class_label", value=label),
            score=probability,
        )
        for label, probability in list(zip(prediction, confidence))
    ]

    return [
        PredictedInstanceSegmentation(
            mask=mask,
            scored_labels=scored_label,
            image=Image(
                uid=image_uid,
                height=image_height,
                width=image_width,
            ),
        )
        for mask, scored_label in list(zip(masks, scored_labels))
    ]


def parse_object_detection(result):

    # Extract data
    # image_uid = Path(result.path).stem
    # image_height = result.orig_shape[0]
    # image_width = result.orig_shape[1]

    # for mem in dir(result.boxes):
    #     print(mem)
    # print(result.boxes)

    pass


def upload_inferences(results: List[dict]):
    for result in results:

        if "masks" in result.keys and "boxes" in result.keys:
            parse_image_segmentation(result=result)
        elif "boxes" in result.keys:
            parse_object_detection(result=result)
        elif "probs" in result.keys:
            parse_image_classification(result=result)

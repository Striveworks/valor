from pathlib import Path
from typing import Union

import numpy
import PIL
from PIL.Image import Resampling

from velour.client import Client, ImageModel
from velour.data_types import (
    BoundingBox,
    Image,
    Label,
    PredictedDetection,
    PredictedImageClassification,
    PredictedInstanceSegmentation,
    ScoredLabel,
)


def parse_yolo_image_classification(result, uid: str):
    """Parses Ultralytic's result for an image classification task."""

    # Extract data
    image_uid = uid
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

    return [
        PredictedImageClassification(
            image=Image(
                uid=image_uid,
                height=image_height,
                width=image_width,
            ),
            scored_labels=scored_labels,
        )
    ]


def _convert_yolo_segmentation(
    raw,
    height: int,
    width: int,
    resample: Resampling = Resampling.BILINEAR,
):
    """Resizes the raw binary mask provided by the YOLO inference to the original image size."""
    mask = numpy.asarray(raw.cpu())
    mask[mask == 1.0] = 255
    img = PIL.Image.fromarray(numpy.uint8(mask))
    img = img.resize((width, height), resample=resample)
    mask = numpy.array(img, dtype=numpy.uint8) >= 128
    return mask


def parse_yolo_image_segmentation(
    result, uid: str, resample: Resampling = Resampling.BILINEAR
):
    """Parses Ultralytic's result for an image segmentation task."""

    # Extract data
    image_uid = uid
    image_height = result.orig_shape[0]
    image_width = result.orig_shape[1]
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    masks = [mask for mask in result.masks.data]

    # Create scored label list
    scored_labels = [
        ScoredLabel(
            label=Label(key="class_label", value=label),
            score=probability,
        )
        for label, probability in list(zip(labels, probabilities))
    ]

    # Extract masks
    masks = [
        _convert_yolo_segmentation(
            raw, height=image_height, width=image_width, resample=resample
        )
        for raw in result.masks.data
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


def parse_yolo_object_detection(result, uid: str):
    """Parses Ultralytic's result for an object detection task."""

    # Extract data
    image_uid = uid
    image_height = result.orig_shape[0]
    image_width = result.orig_shape[1]
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    bboxes = [numpy.asarray(box.cpu()) for box in result.boxes.xyxy]

    # Create scored label list
    scored_labels = [
        ScoredLabel(
            label=Label(key="class_label", value=label),
            score=probability,
        )
        for label, probability in list(zip(labels, probabilities))
    ]

    # Extract Bounding Boxes
    bboxes = [
        BoundingBox(
            xmin=box[0],
            ymin=box[1],
            xmax=box[2],
            ymax=box[3],
        )
        for box in bboxes
    ]

    return [
        PredictedDetection(
            scored_labels=scored_label,
            image=Image(
                uid=image_uid,
                height=image_height,
                width=image_width,
            ),
            bbox=bbox,
        )
        for bbox, scored_label in list(zip(bboxes, scored_labels))
    ]


def parse_yolo_results(
    results,
    uid: str,
    segmentation_resample: Resampling = Resampling.BILINEAR,
) -> Union[
    PredictedDetection,
    PredictedImageClassification,
    PredictedInstanceSegmentation,
]:
    """Automatically chooses correct parser for Ultralytic YOLO model inferences.

    Parameters
    ----------
    result
        YOLO Model Result
    uid
        Image uid

    Returns
    -------
    Velour prediction.
    """

    if "masks" in results.keys and "boxes" in results.keys:
        return parse_yolo_image_segmentation(
            results, uid=uid, resample=segmentation_resample
        )
    elif "boxes" in results.keys:
        return parse_yolo_object_detection(results, uid)
    elif "probs" in results.keys:
        return parse_yolo_image_classification(results, uid)
    else:
        raise ValueError(
            "Input arguement 'result' does not contain identifiable information."
        )


def create_model_from_yolo(
    client: Client, model, name: str = None, description: str = None
) -> ImageModel:
    """Converts yolo model to a velour model.

    Parameters
    ----------
    client
        Velour client object
    model
        YOLO model object
    name
        (OPTIONAL) Defaults to YOLO model name.
    description
        (OPTIONAL) Defaults to YOLO model description.

    Returns
    -------
    Velour image model
    """

    if name is None:
        # Strip model name from yaml path
        path = model.model.yaml["yaml_file"]
        name = Path(path).stem

    return client.create_image_model(name, href=None, description=description)

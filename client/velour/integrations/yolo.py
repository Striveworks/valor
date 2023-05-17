from pathlib import Path
from typing import List

import numpy
import PIL
import torch
from PIL.Image import Resampling

from velour.client import Client, Dataset, Model
from velour.data_types import (
    BoundingBox,
    Image,
    Label,
    PredictedDetection,
    PredictedImageClassification,
    PredictedInstanceSegmentation,
    ScoredLabel,
)

try:
    from ultralytics.yolo.engine.results import Results
except ModuleNotFoundError:
    "'ultralytics' package not found. You can find the project at https://github.com/ultralytics/ultralytics"


def parse_image_classification(result: Results):
    """Parses Ultralytic's results for an image classification task."""

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
    raw: torch.Tensor,
    height: int,
    width: int,
    resample: Resampling = Resampling.BILINEAR,
):
    """Resizes the raw binary mask provided by the YOLO inference to the original image size."""
    mask = raw.cpu().numpy()
    mask[mask == 1.0] = 255
    img = PIL.Image.fromarray(numpy.uint8(mask))
    img = img.resize((width, height), resample=resample)
    mask = numpy.array(img, dtype=numpy.uint8) >= 128
    return mask


def parse_image_segmentation(
    result: Results, resample: Resampling = Resampling.BILINEAR
):
    """Parses Ultralytic's results for an image segmentation task."""

    # Extract data
    image_uid = Path(result.path).stem
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


def parse_object_detection(result: Results):
    """Parses Ultralytic's results for an object detection task."""

    # Extract data
    image_uid = Path(result.path).stem
    image_height = result.orig_shape[0]
    image_width = result.orig_shape[1]
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    bboxes = [box.cpu().numpy() for box in result.boxes.xyxy]

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


def upload_inferences(
    client: Client,
    model_name: str,
    dataset: Dataset,
    results: List[Results],
    segmentation_resample: Resampling = Resampling.BILINEAR,
    chunk_size: int = 1000,
    show_progress_bar: bool = True,
) -> Model:
    """Upload Ultralytic's YOLO model inferences to Velour.

    Parameters
    ----------
    client
        Velour Client object.
    model_name
        Model name.
    dataset
        Velour Dataset object.
    results
        YOLO Inferences
    segmentation_resample
        (OPTIONAL) Defaults to Resampling.BILINEAR filter. This is used when resizing
        masks from output size to orginal image size.
    chunk_size
        (OPTIONAL) Defaults to 1000. Chunk_size is the maximum number of elements that are
        uploaded in one call to the backend.
    show_progress_bar
        (OPTIONAL) Defaults to True. Controls whether a tqdm progress bar is displayed
        to show upload progress.

    Returns
    -------
    Velour Model.
    """

    # Parse inferences
    predictions = []
    for result in results:
        if "masks" in result.keys and "boxes" in result.keys:
            predictions += parse_image_segmentation(
                result=result, resample=segmentation_resample
            )
        elif "boxes" in result.keys:
            predictions += parse_object_detection(result)
        elif "probs" in result.keys:
            predictions += parse_image_classification(result)

    # Create & Populate Model
    model = Model(client=client, name=model_name)
    model.add_predictions(
        dataset=dataset,
        predictions=predictions,
        chunk_size=chunk_size,
        show_progress_bar=show_progress_bar,
    )
    model.finalize_inferences()
    return model

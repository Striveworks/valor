import numpy
import PIL
from PIL.Image import Resampling

from valor import Annotation, Datum, Label, Prediction
from valor.metatypes import ImageMetadata
from valor.schemas import Box, Raster


def parse_detection_into_bounding_box(
    result, datum: Datum, label_key: str = "class"
) -> Prediction:
    """Parses Ultralytic's result for an object detection task."""

    # Extract data
    result = result[0]
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    bboxes = [numpy.asarray(box.cpu()) for box in result.boxes.xyxy]

    # validate dimensions
    image_metadata = ImageMetadata(datum)
    if image_metadata.height != result.orig_shape[0]:
        raise RuntimeError
    if image_metadata.width != result.orig_shape[1]:
        raise RuntimeError

    # Create scored label list
    labels = [
        Label(key=label_key, value=label, score=probability)
        for label, probability in list(zip(labels, probabilities))
    ]

    # Extract Bounding Boxes
    bboxes = [
        Box.from_extrema(
            xmin=int(box[0]),
            ymin=int(box[1]),
            xmax=int(box[2]),
            ymax=int(box[3]),
        )
        for box in bboxes
    ]

    return Prediction(
        datum=datum,
        annotations=[
            Annotation(
                labels=[scored_label],
                bounding_box=bbox,
            )
            for bbox, scored_label in list(zip(bboxes, labels))
        ],
    )


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


def parse_detection_into_raster(
    result,
    datum: Datum,
    label_key: str = "class",
    resample: Resampling = Resampling.BILINEAR,
) -> Prediction:
    """Parses Ultralytic's result for an image segmentation task."""

    result = result[0]

    if result.masks is None:
        return Prediction(
            datum=datum,
            annotations=[],
        )

    # Extract data
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    masks = [mask for mask in result.masks.data]

    # validate dimensions
    image_metadata = ImageMetadata(datum)
    if image_metadata.height != result.orig_shape[0]:
        raise RuntimeError
    if image_metadata.width != result.orig_shape[1]:
        raise RuntimeError

    # Create scored label list
    labels = [
        Label(key=label_key, value=label, score=probability)
        for label, probability in list(zip(labels, probabilities))
    ]

    # Extract masks
    masks = [
        _convert_yolo_segmentation(
            raw,
            height=image_metadata.height,
            width=image_metadata.width,
            resample=resample,
        )
        for raw in result.masks.data
    ]

    # create prediction
    return Prediction(
        datum=datum,
        annotations=[
            Annotation(
                labels=[scored_label],
                raster=Raster.from_numpy(mask),
                is_instance=True,
            )
            for mask, scored_label in list(zip(masks, labels))
        ],
    )

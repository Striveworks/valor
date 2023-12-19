import numpy

from velour import Datum, Annotation, Label, Prediction, enums
from velour.metatypes import ImageMetadata
from velour.schemas import BoundingBox


def parse_object_detection(
    result, 
    datum: Datum, 
    label_key: str = "class"
) -> Prediction:
    """Parses Ultralytic's result for an object detection task."""

    # Extract data
    result = result[0]
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    bboxes = [numpy.asarray(box.cpu()) for box in result.boxes.xyxy]


    # validate dimensions
    image_metadata = ImageMetadata.from_datum(datum)
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
        BoundingBox.from_extrema(
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
                task_type=enums.TaskType.DETECTION,
                labels=[scored_label],
                bounding_box=bbox,
            )
            for bbox, scored_label in list(zip(bboxes, labels))
        ],
    )

from velour import Datum, Annotation, Label, Prediction, enums
from velour.metatypes import ImageMetadata


def parse_yolo_image_classification(
    result,
    datum: Datum,
    label_key: str = "class",
) -> Prediction:
    """Parses Ultralytic's result for an image classification task."""

    # Extract data
    result = result[0]
    probabilities = result.probs
    labels = result.names

    # validate dimensions
    image_metadata = ImageMetadata.from_datum(datum)
    if image_metadata.height != result.orig_shape[0]:
        raise RuntimeError
    if image_metadata.width != result.orig_shape[1]:
        raise RuntimeError

    # Create scored label list
    labels = [
        Label(key=label_key, value=labels[key], score=probability.item())
        for key, probability in list(zip(labels, probabilities))
    ]

    # create prediction
    return Prediction(
        datum=datum,
        annotations=[
            Annotation(task_type=enums.TaskType.CLASSIFICATION, labels=labels)
        ],
    )

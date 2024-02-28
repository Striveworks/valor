import random
from typing import cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from valor import (
    Annotation,
    Client,
    Dataset,
    GroundTruth,
    Label,
    Model,
    Prediction,
    enums,
)
from valor.coretypes import Datum
from valor.metatypes import ImageMetadata
from valor.schemas import BoundingBox, Raster


def _sample_without_replacement(array: list, n: int) -> list:
    """Sample from a list without replacement. Used to draw unique IDs from a pre-populated list"""
    random.shuffle(array)
    output = array[:n]
    del array[:n]
    return output


def _generate_mask(
    height: int,
    width: int,
    minimum_mask_percent: float = 0.05,
    maximum_mask_percent: float = 0.4,
) -> NDArray:
    """Generate a random mask for an image with a given height and width"""
    mask_cutoff = random.uniform(minimum_mask_percent, maximum_mask_percent)
    mask = (np.random.random((height, width))) < mask_cutoff

    return mask


def _generate_gt_annotation(
    height: int,
    width: int,
    unique_label_ids: list,
    n_labels: int,
) -> Annotation:
    """Generate an annotation for a given image with a given number of labels"""
    task_types = [
        enums.TaskType.OBJECT_DETECTION,
        enums.TaskType.SEMANTIC_SEGMENTATION,
    ]
    mask = _generate_mask(height=height, width=width)
    raster = Raster.from_numpy(mask)
    bounding_box = _generate_bounding_box(
        max_height=height, max_width=width, is_random=True
    )
    task_type = random.choice(task_types)

    labels = []
    for i in range(n_labels):
        unique_id = _sample_without_replacement(unique_label_ids, 1)[0]
        label = _generate_label(str(unique_id))
        labels.append(label)

    return Annotation(
        task_type=task_type,
        labels=labels,
        raster=raster,
        bounding_box=bounding_box,
    )


def _generate_label(unique_id: str, add_score: bool = False) -> Label:
    """Generate a label given some unique ID"""
    if not add_score:
        return Label(key="k" + unique_id, value="v" + unique_id)
    else:
        return Label(
            key="k" + unique_id,
            value="v" + unique_id,
            score=random.uniform(0, 1),
        )


def _generate_image_metadata(
    unique_id: str,
    min_height: int = 360,
    max_height: int = 640,
    min_width: int = 360,
    max_width: int = 640,
) -> dict:
    """Generate metadata for an image"""
    height = random.randrange(min_height, max_height)
    width = random.randrange(min_width, max_width)

    return {
        "uid": unique_id,
        "height": height,
        "width": width,
    }


def _generate_ground_truth(
    unique_image_id: str,
    n_annotations: int,
    n_labels: int,
) -> GroundTruth:
    """Generate a GroundTruth for an image with the given number of annotations and labels"""

    image_metadata = _generate_image_metadata(unique_id=unique_image_id)
    image_datum = ImageMetadata(
        uid=image_metadata["uid"],
        height=image_metadata["height"],
        width=image_metadata["width"],
    ).to_datum()

    unique_label_ids = list(range(n_annotations * n_labels))

    annotations = [
        _generate_gt_annotation(
            height=image_metadata["height"],
            width=image_metadata["width"],
            n_labels=n_labels,
            unique_label_ids=unique_label_ids,
        )
        for _ in range(n_annotations)
    ]

    gt = GroundTruth(
        datum=image_datum,
        annotations=annotations,
    )

    return gt


def _generate_bounding_box(
    max_height: int, max_width: int, is_random: bool = False
):
    """Generate an arbitrary bounding box"""

    if is_random:
        x_min = int(random.uniform(0, max_width // 2))
        x_max = int(random.uniform(max_width // 2, max_width))
        y_min = int(random.uniform(0, max_height // 2))
        y_max = int(random.uniform(max_height // 2, max_height))
    else:
        # use the whole image as the bounding box to ensure that we have predictions overlap with groundtruths
        x_min = 0
        x_max = max_width
        y_min = 0
        y_max = max_height

    return BoundingBox.from_extrema(
        xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max
    )


def _generate_prediction_annotation(
    height: int, width: int, unique_label_ids: list, n_labels: int
):
    """Generate an arbitrary inference annotation"""
    box = _generate_bounding_box(max_height=height, max_width=width)
    labels = []
    for i in range(n_labels):
        unique_id = _sample_without_replacement(unique_label_ids, 1)[0]
        label = _generate_label(str(unique_id), add_score=True)
        labels.append(label)

    return Annotation(
        task_type=enums.TaskType.OBJECT_DETECTION,
        labels=labels,
        bounding_box=box,
    )


def _generate_prediction(
    datum: Datum,
    height: int,
    width: int,
    n_annotations: int,
    n_labels: int,
):
    """Generate an arbitrary prediction based on some image"""

    # ensure that some labels are common
    n_label_ids = n_annotations * n_labels
    unique_label_ids = list(range(n_label_ids))

    annotations = [
        _generate_prediction_annotation(
            height=height,
            width=width,
            unique_label_ids=unique_label_ids,
            n_labels=n_labels,
        )
        for _ in range(n_annotations)
    ]

    return Prediction(datum=datum, annotations=annotations)


def generate_segmentation_data(
    client: Client,
    dataset_name: str,
    n_images: int = 10,
    n_annotations: int = 10,
    n_labels: int = 2,
) -> Dataset:
    """
    Generate a synthetic Valor dataset given a set of input images.

    Parameters
    ----------
    client : Session
        The Client object used to access your valor instance.
    dataset_name : str
        The name of the dataset you want to generate in Valor.
    n_images : int
        The number of images you'd like your dataset to contain.
    n_annotations : int
        The number of annotations per image you'd like your dataset to contain.
    n_labels : int
        The number of labels per annotation you'd like your dataset to contain.
    """
    dataset = Dataset.create(dataset_name)

    unique_image_ids = list(range(n_images))
    for _ in tqdm(range(n_images)):
        gt = _generate_ground_truth(
            unique_image_id=str(
                _sample_without_replacement(unique_image_ids, 1)[0]
            ),
            n_annotations=n_annotations,
            n_labels=n_labels,
        )
        dataset.add_groundtruth(gt)

    dataset.finalize()

    return dataset


def generate_prediction_data(
    client: Client,
    dataset: Dataset,
    model_name: str,
    n_annotations: int = 10,
    n_labels: int = 2,
):
    """
    Generate an arbitrary number of predictions for a previously generated dataset.

    Parameters
    ----------
    client : Session
    The Client object used to access your Valor instance.
    dataset : Dataset
        The dataset object to create predictions for.
    model_name : str
        The name of your model.
    n_annotations : int
        The number of annotations per prediction you'd like your dataset to contain.
    n_labels : int
        The number of labels per annotation you'd like your dataset to contain.
    """
    model = Model.create(model_name)

    datums = dataset.get_datums()

    for datum in datums:
        height = cast(int, datum.metadata["height"])
        width = cast(int, datum.metadata["width"])
        prediction = _generate_prediction(
            datum=datum,
            height=int(height),
            width=int(width),
            n_annotations=n_annotations,
            n_labels=n_labels,
        )
        model.add_prediction(dataset, prediction)

    model.finalize_inferences(dataset)
    return model
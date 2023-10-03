import random
from typing import Tuple

import numpy as np
from tqdm import tqdm

from velour import enums
from velour.client import Client, ClientException
from velour.client import Dataset as VelourDataset
from velour.schemas import (
    Annotation,
    GroundTruth,
    ImageMetadata,
    Label,
    Raster,
)


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
) -> np.array:
    """Generate a random mask for an image with a given height and width"""
    mask_cutoff = random.uniform(minimum_mask_percent, maximum_mask_percent)
    mask = (np.random.random((height, width))) < mask_cutoff

    return mask


def _generate_annotation(
    height: int,
    width: int,
    unique_label_ids: list,
    n_labels: int,
) -> Annotation:
    """Generate an annotation for a given image with a given number of labels"""
    task_types = [
        enums.TaskType.INSTANCE_SEGMENTATION,
        enums.TaskType.SEMANTIC_SEGMENTATION,
    ]
    mask = _generate_mask(height=height, width=width)
    raster = Raster.from_numpy(mask)
    task_type = random.choice(task_types)

    labels = []
    for i in range(n_labels):
        unique_id = _sample_without_replacement(unique_label_ids, 1)[0]
        label = _generate_label(str(unique_id))
        labels.append(label)

    return Annotation(task_type=task_type, labels=labels, raster=raster)


def _generate_label(unique_id: str) -> Label:
    """Generate a label given some unique ID"""
    return Label(key="k" + unique_id, value="v" + unique_id)


def _generate_image_metadata(
    unique_id: int,
    min_height: int = 360,
    max_height: int = 640,
    min_width: int = 360,
    max_width: int = 640,
) -> Tuple[ImageMetadata, int, int]:
    height = random.randrange(min_height, max_height)
    width = random.randrange(min_width, max_width)

    return {
        "uid": str(unique_id),
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
        _generate_annotation(
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


def generate_segmentation_data(
    client: Client,
    dataset_name: str,
    n_images: int = 10,
    n_annotations: int = 10,
    n_labels: int = 2,
) -> VelourDataset:
    """
    Generate a synthetic velour dataset given a set of input images

    Parameters
    ----------
    client
        The Client object used to access your velour instance
    dataset_name
        The name of the dataset you want to generate in velour
    n_images
        The number of images you'd like your dataset to contain
    n_annotations
        The number of annotations per image you'd like your dataset to contain
    n_labels
        The number of labels per annotation you'd like your dataset to contain
    """

    try:
        client.delete_dataset(dataset_name)
        dataset = VelourDataset.create(client, dataset_name)
    except ClientException:
        dataset = VelourDataset.get(client, dataset_name)

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

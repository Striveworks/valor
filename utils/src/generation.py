import json
import random

import numpy as np
from tqdm import tqdm

from velour import enums
from velour.client import ClientException
from velour.client import Dataset as VelourDataset
from velour.schemas import (
    Annotation,
    GroundTruth,
    ImageMetadata,
    Label,
    Raster,
)
from velour_api import enums
from velour_api.schemas.geometry import Raster
from velour_api.schemas.label import Label


def _sample_without_replacement(array: list, n: int) -> list:
    """Sample from a list without replacement. Used to draw unique IDs from a pre-populated list"""
    random.shuffle(array)
    output = array[:n]
    del array[:n]
    return output


def _generate_ground_truth(
    image: str, n_annotations: int, n_labels: int, unique_label_ids: list
) -> GroundTruth:
    """Generate a GroundTruth for an image with the given number of annotations and labels"""
    image_id = image["id"]
    image_metadata = ImageMetadata(
        uid=str(image_id),
        height=image["height"],
        width=image["width"],
    ).to_datum()

    annotations = [
        _generate_annotation(
            height=image["height"],
            width=image["width"],
            n_labels=n_labels,
            unique_label_ids=unique_label_ids,
        )
        for i in range(n_annotations)
    ]

    gt = GroundTruth(
        datum=image_metadata,
        annotations=annotations,
    )

    return gt


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


def generate_segmentation_data(
    client: str,
    dataset_name: str,
    metadata_json_path: str,
    n_images: int = 10,
    n_annotations: int = 10,
    n_labels: int = 2,
) -> VelourDataset:
    """
    Generate a synthetic velour dataset given a set of input images

    Parameters
    ----------
    client
        The host to connect to. Should start with "http://" or "https://"
    dataset_name
        The name of the dataset you want to generate in velour
    metadata_json_path
        A JSON which specifies where to find your input images. Should match the format of utils/sample_iamges/sample_coco_images.json
    n_images
        the access token if the host requires authentication
    n_annotations
        the access token if the host requires authentication
    n_labels
        the access token if the host requires authentication
    """
    with open(metadata_json_path) as f:
        metadata = json.load(f)

    n_images_in_metadata = len(metadata["images"])

    if not n_images <= n_images_in_metadata:
        raise ValueError(
            f"n_images must be less than or equal to {n_images_in_metadata}, which is the number of images in your metadata_json_path"
        )

    try:
        client.delete_dataset(dataset_name)
        dataset = VelourDataset.create(client, dataset_name)
    except ClientException:
        dataset = VelourDataset.get(client, dataset_name)

    images = metadata["images"][:n_images]

    for image in tqdm(images):
        # label keys/values can't be shared across images, so
        # we randomly sample them without replacement
        unique_label_ids = list(range(n_annotations * n_labels))

        gt = _generate_ground_truth(
            image=image,
            n_annotations=n_annotations,
            n_labels=n_labels,
            unique_label_ids=unique_label_ids,
        )
        dataset.add_groundtruth(gt)

    dataset.finalize()

    return dataset

import io
import random
from dataclasses import asdict
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    Filter,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import AnnotationType, EvaluationStatus, TaskType
from valor.metatypes import ImageMetadata
from valor.schemas import Box, Raster


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
    mask = _generate_mask(height=height, width=width)
    raster = Raster.from_numpy(mask)
    bounding_box = _generate_bounding_box(
        max_height=height, max_width=width, is_random=True
    )

    labels = []
    for i in range(n_labels):
        unique_id = _sample_without_replacement(unique_label_ids, 1)[0]
        label = _generate_label(str(unique_id))
        labels.append(label)

    return Annotation(
        labels=labels,
        raster=raster,
        bounding_box=(bounding_box),
        is_instance=True,
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
    image_datum = ImageMetadata.create(
        uid=image_metadata["uid"],
        height=image_metadata["height"],
        width=image_metadata["width"],
    ).datum

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

    return Box.from_extrema(xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max)


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
        labels=labels,
        bounding_box=box,
        is_instance=True,
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


def _mask_bytes_to_pil(mask_bytes):
    with io.BytesIO(mask_bytes) as f:
        return Image.open(f)


def test_generate_segmentation_data(
    client: Client,
    dataset_name: str,
    n_images: int = 10,
    n_annotations: int = 2,
    n_labels: int = 2,
):
    """Check that our generated dataset correctly matches our input parameters"""

    dataset = generate_segmentation_data(
        client=client,
        dataset_name=dataset_name,
        n_images=n_images,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    sample_images = dataset.get_datums()
    assert (
        len(sample_images) == n_images
    ), "Number of images doesn't match the test input"

    for image in dataset.get_datums():
        uid = image.uid
        sample_gt = dataset.get_groundtruth(uid)

        assert sample_gt
        sample_annotations = sample_gt.annotations
        assert sample_annotations[0].raster.get_value() is not None
        sample_mask_size = Image.fromarray(
            sample_annotations[0].raster.array
        ).size

        sample_image = ImageMetadata(sample_gt.datum)
        sample_image_size = (sample_image.width, sample_image.height)

        assert (
            len(sample_annotations) == n_annotations
        ), "Number of annotations doesn't match the test input"
        assert (
            len(sample_annotations[0].labels) == n_labels
        ), "Number of labels on the sample annotation doesn't match the test input"
        assert (
            sample_image_size == sample_mask_size
        ), f"Image is size {sample_image_size}, but mask is size {sample_mask_size}"


def test_generate_prediction_data(client: Client):
    """Check that our generated predictions correctly matches our input parameters"""

    n_images = 10
    dataset_name = "dset"
    model_name = "model"

    dataset = generate_segmentation_data(
        client=client,
        dataset_name=dataset_name,
        n_images=n_images,
        n_annotations=10,
        n_labels=10,
    )
    assert len(dataset.get_datums()) == n_images

    model = generate_prediction_data(
        client=client,
        dataset=dataset,
        model_name=model_name,
        n_annotations=5,
        n_labels=5,
    )

    eval_job = model.evaluate_detection(
        dataset,
        iou_thresholds_to_compute=[0.1, 0.9],
        iou_thresholds_to_return=[0.1, 0.9],
        filter_by=[Label.key == "k1"],
        convert_annotations_to_type=AnnotationType.BOX,
    )
    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    eval_dict = eval_job.to_dict()
    for key in [
        "id",
        "confusion_matrices",
        "metrics",
        "status",
        "ignored_pred_labels",
        "missing_pred_labels",
    ]:
        eval_dict.pop(key)

    # check meta separately since duration isn't deterministic
    assert eval_dict["meta"]["datums"] == 10
    assert (
        eval_dict["meta"]["labels"] == 1
    )  # we're filtering on one label above
    assert eval_dict["meta"]["duration"] <= 30
    eval_dict["meta"] = {}

    assert eval_dict == {
        "model_name": model_name,
        "datum_filter": {
            **asdict(
                Filter()
            ),  # default filter properties with overrides below
            "dataset_names": [dataset_name],
            "label_keys": ["k1"],
        },
        "parameters": {
            "task_type": TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": AnnotationType.BOX.value,
            "iou_thresholds_to_compute": [0.1, 0.9],
            "iou_thresholds_to_return": [0.1, 0.9],
            "label_map": None,
            "recall_score_threshold": 0.0,
            "metrics_to_return": [
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
        },
        "meta": {},
    }
    assert len(eval_job.metrics) > 0

import gzip
import json
import tempfile
import urllib
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

from velour.client import Client, ImageDataset, ImageModel
from velour.data_types import (
    BoundingBox,
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    GroundTruthSemanticSegmentation,
    Image,
    Label,
    Point,
    PolygonWithHole,
    PredictedDetection,
    ScoredLabel,
)

try:
    import chariot
    from chariot.config import settings
    from chariot.datasets.dataset_version import (
        get_latest_vertical_dataset_version,
    )

except ModuleNotFoundError:
    "`chariot` package not found. if you have an account on Chariot please see https://production.chariot.striveworks.us/docs/sdk/sdk for how to install the python SDK"


def _retrieve_chariot_annotations(manifest_url: str):
    """Retrieves and unpacks Chariot dataset annotations from a manifest url."""

    chariot_dataset = []

    # Create a temporary file
    with tempfile.TemporaryFile(mode="w+b") as f:

        # Download compressed jsonl file
        response = requests.get(manifest_url, stream=True)

        # Progress bar
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Download Chariot Manifest",
        )

        # Write to tempfile if status ok
        if response.status_code == 200:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
            f.flush()
            f.seek(0)
        progress_bar.close()

        # Unzip
        gzf = gzip.GzipFile(mode="rb", fileobj=f)
        jsonl = gzf.read().decode().strip().split("\n")

        # Parse into list of json object(s)
        for line in jsonl:
            chariot_dataset.append(json.loads(line))

    return chariot_dataset


def parse_image_classification(
    datum: dict,
) -> GroundTruthImageClassification:
    """Parses Chariot image classification annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    gt_dets = []
    for annotation in datum["annotations"]:
        gt_dets.append(
            GroundTruthImageClassification(
                image=Image(
                    uid=uid,
                    height=-1,
                    width=-1,
                ),
                labels=[
                    Label(key="class_label", value=annotation["class_label"])
                ],
            )
        )
    return gt_dets


def parse_image_segmentation(
    datum: dict,
) -> GroundTruthSemanticSegmentation:
    """Parses Chariot image segmentation annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    annotated_regions = {}
    for annotation in datum["annotations"]:

        annotation_label = annotation["class_label"]

        hole = None

        # Create BoundingPolygon
        points = [
            Point(x=point["x"], y=point["y"])
            for point in annotation["contours"][0]
        ]
        polygon = BoundingPolygon(points)

        # Create BoundingPolygon if hole exists
        hole = None
        if len(annotation["contours"]) > 1:
            points = [
                Point(x=point["x"], y=point["y"])
                for point in annotation["contours"][1]
            ]
            hole = BoundingPolygon(points)

        # Populate PolygonWithHole
        region = PolygonWithHole(polygon=polygon, hole=hole)

        # Add annotated region to list
        if region is not None:
            if annotation_label not in annotated_regions:
                annotated_regions[annotation_label] = []
            annotated_regions[annotation_label].append(region)

    # Create a list of GroundTruths
    gt_dets = []
    for label in annotated_regions.keys():
        gt_dets.append(
            GroundTruthSemanticSegmentation(
                shape=annotated_regions[label],
                labels=[Label(key="class_label", value=label)],
                image=Image(
                    uid=uid,
                    height=-1,
                    width=-1,
                ),
            )
        )
    return gt_dets


def parse_object_detection(
    datum: dict,
) -> GroundTruthDetection:
    """Parses Chariot object detection annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    gt_dets = []
    for annotation in datum["annotations"]:
        gt_dets.append(
            GroundTruthDetection(
                bbox=BoundingBox(
                    xmin=annotation["bbox"]["xmin"],
                    ymin=annotation["bbox"]["ymin"],
                    xmax=annotation["bbox"]["xmax"],
                    ymax=annotation["bbox"]["ymax"],
                ),
                labels=[
                    Label(key="class_label", value=annotation["class_label"])
                ],
                image=Image(
                    uid=uid,
                    height=-1,
                    width=-1,
                ),
            )
        )
    return gt_dets


def _parse_chariot_annotations(
    chariot_manifest, chariot_task_type, use_training_manifest: bool = True
) -> list:
    """Get chariot dataset annotations.

    Parameters
    ----------
    chariot_manifest
        List of dictionaries containing data annotations and metadata.
    chariot_task_type
        Pass-through property chariot.datasets.dataset_version.DatasetVersion.supported_task_types
    using_training_manifest
        (OPTIONAL) Defaults to true, setting false will use the evaluation manifest which is a
        super set of the training manifest. Not recommended as the evaluation manifest may
        contain unlabeled data.

    Returns
    -------
    Chariot annotations in velour 'groundtruth' format.
    """

    # Image Classification
    if chariot_task_type.image_classification:
        groundtruth_annotations = [
            gt
            for datum in chariot_manifest
            for gt in parse_image_classification(datum)
        ]

    # Image Segmentation
    elif chariot_task_type.image_segmentation:
        groundtruth_annotations = [
            gt
            for datum in chariot_manifest
            for gt in parse_image_segmentation(datum)
        ]

    # Object Detection
    elif chariot_task_type.object_detection:
        groundtruth_annotations = [
            gt
            for datum in chariot_manifest
            for gt in parse_object_detection(datum)
        ]

    # Text Sentiment
    elif chariot_task_type.text_sentiment:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    # Text Summarization
    elif chariot_task_type.text_summarization:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    # Text Token Classifier
    elif chariot_task_type.text_token_classification:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    # Text Translation
    elif chariot_task_type.text_translation:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    return groundtruth_annotations


def upload_dataset(
    client: Client,
    dataset: chariot.datasets.dataset.Dataset,
    dataset_version_id: str = None,
    name: str = None,
    use_training_manifest: bool = True,
    chunk_size: int = 1000,
    show_progress_bar: bool = True,
):
    """Converts chariot dataset to a velour dataset.

    Parameters
    ----------
    client
        Velour client object
    dataset
        Chariot Dataset object
    dataset_version_id
        (OPTIONAL) Chariot Dataset version ID, defaults to latest vertical version.
    name
        (OPTIONAL) Defaults to the name of the chariot dataset, setting this will override the
        name of the velour dataset.
    using_training_manifest
        (OPTIONAL) Defaults to true, setting false will use the evaluation manifest which is a
        super set of the training manifest. Not recommended as the evaluation manifest may
        contain unlabeled data.
    chunk_size
        (OPTIONAL) Defaults to 1000. Chunk_size is the maximum number of 'groundtruths' that are
        uploaded in one call to the backend.
    show_progress_bar
        (OPTIONAL) Defaults to True. Controls whether a tqdm progress bar is displayed
        to show upload progress.

    Returns
    -------
    Velour image dataset
    """

    if len(dataset.versions) < 1:
        raise ValueError("Chariot dataset has no existing versions.")

    # Get Chariot Datset Version
    dsv = None
    if dataset_version_id is None:
        # Use the latest version
        dsv = get_latest_vertical_dataset_version(
            project_id=dataset.project_id,
            dataset_id=dataset.id,
        )
    else:
        for _dsv in dataset.versions:
            if _dsv.id == dataset_version_id:
                dsv = _dsv
                break
        if dsv is None:
            raise ValueError(
                f"Chariot DatasetVersion not found with id: {dataset_version_id}"
            )

    # Get the manifest url
    manifest_url = (
        dsv.get_training_manifest_url()
        if use_training_manifest
        else dsv.get_evaluation_manifest_url()
    )

    # Retrieve the manifest
    chariot_annotations = _retrieve_chariot_annotations(manifest_url)

    # Get GroundTruth Annotations
    groundtruth_annotations = _parse_chariot_annotations(
        chariot_annotations,
        dsv.supported_task_types,
        use_training_manifest,
    )

    # Check if name has been overwritten
    if name is None:
        name = dataset.name

    # Construct url
    href = urllib.parse.urljoin(
        settings.base_url,
        f"/projects/{dsv.project_id}/datasets/{dsv.dataset_id}",
    )

    # Create velour dataset
    velour_dataset = client.create_dataset(name=name, href=href)

    # Upload velour dataset
    velour_dataset.add_groundtruth(
        groundtruth_annotations,
        chunk_size=chunk_size,
    )

    # Finalize and return
    velour_dataset.finalize()
    return velour_dataset


def _parse_chariot_detections(
    dets: dict,
    image: Image,
    label_key: str = "class",
) -> List[PredictedDetection]:

    expected_keys = {
        "num_detections",
        "detection_classes",
        "detection_boxes",
        "detection_scores",
    }

    if set(dets.keys()) != expected_keys:
        raise ValueError(
            f"Expected `dets` to have keys {expected_keys} but got {dets.keys()}"
        )

    return [
        PredictedDetection(
            bbox=BoundingBox(
                ymin=box[0], xmin=box[1], ymax=box[2], xmax=box[3]
            ),
            scored_labels=[
                ScoredLabel(
                    label=Label(key=label_key, value=label), score=score
                )
            ],
            image=image,
        )
        for box, score, label in zip(
            dets["detection_boxes"],
            dets["detection_scores"],
            dets["detection_classes"],
        )
    ]


def upload_inferences(
    client: Client,
    model_name: str,
    dataset: ImageDataset,
    dets: dict,
    image: Image,
    label_key: str = "class",
    chunk_size: int = 1000,
    show_progress_bar: bool = True,
) -> ImageModel:
    """Converts the outputs of a Chariot detection model to velour's format

    Parameters
    ----------
    client
        Velour client object
    dataset
        Velour Dataset object
    model_name:
        Name of the model used to generate these inferences.
    dets:
        Chariot detections.
    image:
        Velour Image object.
    label_key:
        (OPTIONAL) Defaults to 'class'. Key to class label of inference.
    chunk_size
        (OPTIONAL) Defaults to 1000. Chunk_size is the maximum number of 'groundtruths' that are
        uploaded in one call to the backend.
    show_progress_bar
        (OPTIONAL) Defaults to True. Controls whether a tqdm progress bar is displayed
        to show upload progress.

    Returns
    -------
    Velour image model.
    """

    predictions = _parse_chariot_detections(dets, image, label_key)

    # Create & Populate Model
    model = ImageModel(client=client, name=model_name)
    model.add_predictions(
        dataset=dataset,
        predictions=predictions,
        chunk_size=chunk_size,
        show_progress_bar=show_progress_bar,
    )
    model.finalize_inferences()
    return model

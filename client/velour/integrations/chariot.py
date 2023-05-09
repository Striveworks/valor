import gzip
import json
import tempfile
from pathlib import Path

import requests
from chariot.config import settings
from chariot.datasets.dataset import Dataset
from chariot.datasets.dataset_version import DatasetVersion

from velour.client import Client
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
)

# REMOVE
placeholder_image_length = -1


def retrieve_chariot_manifest(manifest_url: str):
    """Retrieves and unpacks Chariot dataset annotations from a manifest url."""

    chariot_dataset = []

    # Create a temporary file
    with tempfile.TemporaryFile(mode="w+b") as f:

        # Download compressed jsonl file
        response = requests.get(manifest_url, stream=True)
        if response.status_code == 200:
            f.write(response.raw.read())
        f.flush()
        f.seek(0)

        # Unzip
        gzf = gzip.GzipFile(mode="rb", fileobj=f)
        jsonl = gzf.read().decode().strip().split("\n")

        # Parse into list of json object(s)
        for line in jsonl:
            chariot_dataset.append(json.loads(line))

    return chariot_dataset


def chariot_parse_image_classification_annotation(
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
                    height=placeholder_image_length,
                    width=placeholder_image_length,
                ),
                labels=[
                    Label(key="class_label", value=annotation["class_label"])
                ],
            )
        )
    return gt_dets


def chariot_parse_image_segmentation_annotation(
    datum: dict,
) -> GroundTruthSemanticSegmentation:
    """Parses Chariot image segmentation annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    annotated_regions = {}
    for annotation in datum["annotations"]:

        annotation_label = annotation["class_label"]

        # Create PolygonWithHole
        region = None

        # Contour name changes based on example...
        contour_key = "contours"
        if "contours" not in annotation and "contour" in annotation:
            contour_key = "contour"
        elif "contours" not in annotation:
            raise ("Missing contour key!")

        polygon = None
        hole = None

        # Create Bounding Polygon
        points = []
        for point in annotation[contour_key][0]:
            points.append(
                Point(
                    x=point["x"],
                    y=point["y"],
                )
            )
        polygon = BoundingPolygon(points)

        # Check if hole exists
        if len(annotation[contour_key]) > 1:
            # Create Bounding Polygon for Hole
            points = []
            for point in annotation[contour_key][1]:
                points.append(
                    Point(
                        x=point["x"],
                        y=point["y"],
                    )
                )
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
                    height=placeholder_image_length,
                    width=placeholder_image_length,
                ),
            )
        )
    return gt_dets


def chariot_parse_object_detection_annotation(
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
                    height=placeholder_image_length,
                    width=placeholder_image_length,
                ),
            )
        )
    return gt_dets


def chariot_parse_dataset_version_manifest(
    chariot_dataset_version: DatasetVersion, use_training_manifest: bool = True
) -> list:
    """Get chariot dataset annotations.

    Parameters
    ----------
    chariot_dataset_version
        Chariot DatasetVersion object of interest.
    using_training_manifest
        (OPTIONAL) Defaults to true, setting false will use the evaluation manifest which is a
        super set of the training manifest. Not recommended as the evaluation manifest may
        contain unlabeled data.

    Returns
    -------
    Chariot annotations in velour format.
    """

    # Get the manifest url
    manifest_url = (
        chariot_dataset_version.get_training_manifest_url()
        if use_training_manifest
        else chariot_dataset_version.get_evaluation_manifest_url()
    )

    # Retrieve the manifest
    chariot_manifest = retrieve_chariot_manifest(manifest_url)

    # Iterate through each element in the dataset
    groundtruth_annotations = []
    for datum in chariot_manifest:

        # Image Classification
        if chariot_dataset_version.supported_task_types.image_classification:
            groundtruth_annotations += (
                chariot_parse_image_classification_annotation(datum)
            )

        # Image Segmentation
        if chariot_dataset_version.supported_task_types.image_segmentation:
            groundtruth_annotations += (
                chariot_parse_image_segmentation_annotation(datum)
            )

        # Object Detection
        if chariot_dataset_version.supported_task_types.object_detection:
            groundtruth_annotations += (
                chariot_parse_object_detection_annotation(datum)
            )

        # Text Sentiment
        if chariot_dataset_version.supported_task_types.text_sentiment:
            pass

        # Text Summarization
        if chariot_dataset_version.supported_task_types.text_summarization:
            pass

        # Text Token Classifier
        if (
            chariot_dataset_version.supported_task_types.text_token_classification
        ):
            pass

        # Text Translation
        if chariot_dataset_version.supported_task_types.text_translation:
            pass

    return groundtruth_annotations


def chariot_ds_to_velour_ds(
    velour_client: Client,
    chariot_dataset: Dataset,
    chariot_dataset_version: str = None,
    velour_dataset_name: str = None,
    use_training_manifest: bool = True,
):
    """Converts chariot dataset to a velour dataset.

    Parameters
    ----------
    velour_client
        Velour client
    chariot_dataset
        Chariot Dataset object
    chariot_dataset_version
        (OPTIONAL) Chariot Dataset version ID, defaults to latest.
    velour_dataset_name
        (OPTIONAL) Defaults to the name of the chariot dataset, setting this will override the
        name of the velour dataset output.
    using_training_manifest
        (OPTIONAL) Defaults to true, setting false will use the evaluation manifest which is a
        super set of the training manifest. Not recommended as the evaluation manifest may
        contain unlabeled data.

    Returns
    -------
    Velour dataset
    """

    if len(chariot_dataset.versions) < 1:
        raise ValueError("Chariot Dataset has no existing versions!")

    # Get Chariot Datset Version
    dsv = None
    if chariot_dataset_version is not None:
        # Attempt to find requested version
        for datasetversion in chariot_dataset.versions:
            if datasetversion.id == chariot_dataset_version:
                dsv = datasetversion
                break
        if dsv is None:
            raise ValueError(
                "Chariot dataset does not have specified version!",
                chariot_dataset_version,
            )
    else:
        # Use the first version in the list
        dsv = chariot_dataset.versions[0]

    # Get GroundTruth Annotations
    groundtruth_annotations = chariot_parse_dataset_version_manifest(
        dsv, use_training_manifest
    )

    # Check if name has been overwritten
    if velour_dataset_name is None:
        velour_dataset_name = chariot_dataset.name

    # Construct url
    href = settings.base_url
    href += "/projects/" + dsv.project_id
    href += "/datasets/" + dsv.dataset_id
    # href += dsv.id

    # Create Velour dataset
    velour_dataset = velour_client.create_dataset(
        name=velour_dataset_name, href=href
    )
    velour_dataset.add_groundtruth_classifications(groundtruth_annotations)
    velour_dataset.finalize()

    return velour_dataset

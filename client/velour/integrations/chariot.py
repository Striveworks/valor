import gzip
import json
import tempfile
from pathlib import Path

import requests
from chariot.config import settings
from chariot.datasets.dataset import Dataset

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
                image=Image(uid=uid, height=None, width=None),
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
                image=Image(uid=uid, height=None, width=None),
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
                image=Image(uid=uid, height=None, width=None),
            )
        )
    return gt_dets


def chariot_parse_text_sentiment_annotation(datum: dict):
    """Parses Chariot text sentiment annotation."""
    return None


def chariot_parse_text_summarization_annotation(datum: dict):
    """Parses Chariot text summarization annotation."""
    return None


def chariot_parse_text_token_classification_annotation(datum: dict):
    """Parses Chariot text token classification annotation."""
    return None


def chariot_parse_text_translation_annotation(datum: dict):
    """Parses Chariot text translation annotation."""
    return None


def chariot_ds_to_velour_ds(
    client: Client,
    chariot_ds: Dataset,
    chariot_ds_version: str = None,
    velour_ds_name: str = None,
    use_training_manifest: bool = True,
):
    """Converts the annotations of a Chariot dataset to velour's format.

    Parameters
    ----------
    client
        Velour client
    chariot_ds
        Chariot Dataset object
    chariot_ds_version
        (OPTIONAL) Chariot Dataset version ID, defaults to latest.
    velour_ds_name
        (OPTIONAL) Defaults to the name of the chariot dataset, setting this will override the
        name of the velour dataset output.
    using_training_manifest
        (OPTIONAL) Defaults to true, setting false will use the evaluation manifest which is a
        super set of the training manifest. Not recommended as the evaluation manifest may
        contain unlabeled data.

    Returns
    -------
    List of Ground Truth Detections
    """

    if len(chariot_ds.versions) < 1:
        raise ValueError("Chariot Dataset has no existing versions!")

    # Get Chariot Datset Version
    dsv = None
    if chariot_ds_version is not None:
        # Attempt to find requested version
        for datasetversion in chariot_ds.versions:
            if datasetversion.id == chariot_ds_version:
                dsv = datasetversion
                break
        if dsv is None:
            raise ValueError(
                "Chariot Dataset does not have specified version!",
                chariot_ds_version,
            )
    else:
        # Use the first version in the list
        dsv = chariot_ds.versions[0]

    # Retrieve the manifest url
    manifest_url = (
        dsv.get_training_manifest_url()
        if use_training_manifest
        else dsv.get_evaluation_manifest_url()
    )

    # Retrieve the manifest
    chariot_manifest = retrieve_chariot_manifest(manifest_url)
    gt_dets = []

    # Iterate through each element in the dataset
    for datum in chariot_manifest:

        # Image Classification
        if dsv.supported_task_types.image_classification:
            gt_dets += chariot_parse_image_classification_annotation(datum)

        # Image Segmentation
        if dsv.supported_task_types.image_segmentation:
            gt_dets += chariot_parse_image_segmentation_annotation(datum)

        # Object Detection
        if dsv.supported_task_types.object_detection:
            gt_dets += chariot_parse_object_detection_annotation(datum)

        # Text Sentiment
        if dsv.supported_task_types.text_sentiment:
            pass

        # Text Summarization
        if dsv.supported_task_types.text_summarization:
            pass

        # Text Token Classifier
        if dsv.supported_task_types.text_token_classification:
            pass

        # Text Translation
        if dsv.supported_task_types.text_translation:
            pass

    # Check if name has been overwritten
    if velour_ds_name is None:
        velour_ds_name = chariot_ds.name

    # Construct url
    href = settings.base_url
    href += "/projects/" + dsv.project_id
    href += "/datasets/" + dsv.dataset_id
    # href += dsv.id

    # Create Velour dataset
    velour_ds = client.create_dataset(name=velour_ds_name, href=href)
    velour_ds.add_groundtruth_classifications(gt_dets)
    velour_ds.finalize()

    return velour_ds

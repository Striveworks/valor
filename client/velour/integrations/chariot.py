import gzip
import json
import tempfile
import urllib
from pathlib import Path
from typing import Dict, List, Union
from dataclasses import dataclass

import requests
from tqdm import tqdm

from velour import enums
from velour.client import Client, Dataset, Model
from velour.schemas import (
    Annotation,
    BasicPolygon,
    BoundingBox,
    GroundTruth,
    Image,
    Label,
    Point,
    Polygon,
    Prediction,
    ScoredAnnotation,
    ScoredLabel,
    MetaDatum,
)

try:
    import chariot
    from chariot.datasets import Dataset as ChariotDataset
    from chariot.models import Model as ChariotModel
    from chariot.models import TaskType as ChariotTaskType
except ModuleNotFoundError:
    "`chariot` package not found. if you have an account on Chariot please see https://production.chariot.striveworks.us/docs/sdk/sdk for how to install the python SDK"


""" Dataset conversion """


def upload_chariot_dataset(       
    client: Client,
    dataset: chariot.datasets.Dataset,
    dataset_version_id: str = None,
) -> Dataset:
    pass


def upload_chariot_model(
    client: Client,
    model: chariot.models.Model,
):
    pass


def _construct_url(
    project_id: str, dataset_id: str = None, model_id: str = None
):

    if dataset_id is not None and model_id is not None:
        raise ValueError("Please provide EITHER model id or dataset id.")
    elif dataset_id is not None:
        href = f"/projects/{project_id}/datasets/{dataset_id}"
    elif model_id is not None:
        href = f"/projects/{project_id}/models/{model_id}"
    else:
        href = f"/projects/{project_id}"

    return urllib.parse.urljoin(
        chariot.config.settings.base_url,
        href,
    )


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


def _parse_image_classification_groundtruths(
    datum: dict,
    label_key: str,
) -> GroundTruth:
    """Parses Chariot image classification annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    image = Image(
        uid=uid,
        height=-1,
        width=-1,
    ).to_datum()

    return GroundTruth(
        datum=image,
        annotations=[
            Annotation(
                task_type=enums.TaskType.CLASSIFICATION,
                labels=[Label(key=label_key, value=annotation["class_label"])],
            )
            for annotation in datum["annotations"]
        ],
    )


def _parse_image_segmentation_groundtruths(
    datum: dict,
    label_key: str,
) -> GroundTruth:
    """Parses Chariot image segmentation annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    image = Image(
        uid=uid,
        height=-1,
        width=-1,
    ).to_datum()

    return GroundTruth(
        datum=image,
        annotations=[
            Annotation(
                task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                labels=[Label(key=label_key, value=annotation["class_label"])],
                polygon=Polygon(
                    boundary=BasicPolygon(
                        points=[
                            Point(x=point["x"], y=point["y"])
                            for point in annotation["contours"][0]
                        ]
                    ),
                    holes=[
                        BasicPolygon(
                            points=[
                                Point(x=point["x"], y=point["y"])
                                for point in annotation["contours"][1]
                            ],
                        )
                    ]
                    if len(annotation["contours"]) > 1
                    else None,
                ),
            )
            for annotation in datum["annotations"]
        ],
    )


def _parse_object_detection_groundtruths(
    datum: dict,
    label_key: str,
) -> GroundTruth:
    """Parses Chariot object detection annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    image = Image(
        uid=uid,
        height=-1,
        width=-1,
    ).to_datum()

    return GroundTruth(
        datum=image,
        annotations=[
            Annotation(
                task_type=enums.TaskType.DETECTION,
                labels=[Label(key=label_key, value=annotation["class_label"])],
                bounding_box=BoundingBox.from_extrema(
                    xmin=annotation["bbox"]["xmin"],
                    ymin=annotation["bbox"]["ymin"],
                    xmax=annotation["bbox"]["xmax"],
                    ymax=annotation["bbox"]["ymax"],
                ),
            )
            for annotation in datum["annotations"]
        ],
    )


def _parse_chariot_groundtruths(
    chariot_manifest,
    chariot_task_type,
    label_key: str,
    use_training_manifest: bool = True,
) -> list:
    """Get chariot dataset annotations.

    Parameters
    ----------
    chariot_manifest
        List of dictionaries containing data annotations and metadata.
    chariot_task_type
        Pass-through property chariot.datasets.DatasetVersion.supported_task_types
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
            _parse_image_classification_groundtruths(
                datum=datum,
                label_key=label_key,
            )
            for datum in chariot_manifest
        ]

    # Image Segmentation
    elif chariot_task_type.image_segmentation:
        groundtruth_annotations = [
            _parse_image_segmentation_groundtruths(
                datum=datum,
                label_key=label_key,
            )
            for datum in chariot_manifest
        ]

    # Object Detection
    elif chariot_task_type.object_detection:
        groundtruth_annotations = [
            _parse_object_detection_groundtruths(
                datum=datum,
                label_key=label_key,
            )
            for datum in chariot_manifest
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


def upload_chariot_dataset(
    client: Client,
    dataset: ChariotDataset,
    dataset_version_id: str = None,
    label_key: str = "class",
    use_training_manifest: bool = False,
    show_progress_bar: bool = False, # Need to implement
) -> Dataset:
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
        dsv = chariot.datasets.get_latest_vertical_dataset_version(
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

    # Get GroundTruths
    groundtruths = _parse_chariot_groundtruths(
        chariot_manifest=chariot_annotations,
        chariot_task_type=dsv.supported_task_types,
        label_key=label_key,
        use_training_manifest=use_training_manifest,
    )

    # Construct url
    href = _construct_url(project_id=dsv.project_id, dataset_id=dsv.dataset_id)

    # Create velour dataset
    velour_dataset = Dataset.create(
        client=client, 
        name=dataset.id,
        href=href,
        description=dsv.summary,
        integration="chariot",
        title=dataset.name,
        project_id=dataset.project_id,
        dataset_version_id=dsv.id,
        task=dsv.type,
    )

    # Upload velour dataset
    for gt in groundtruths:
        velour_dataset.add_groundtruth(gt)

    # Finalize and return
    velour_dataset.finalize()
    return velour_dataset


def parse_chariot_image_classifications(
    classifications: Union[Dict, List[Dict]],
    images: Image,
    label_key: str = "class",
) -> Prediction:
    raise NotImplementedError


def parse_chariot_image_segmentations(
    segmentations: Union[Dict, List[Dict]],
    image: Image,
    label_key: str = "class",
) -> Prediction:
    raise NotImplementedError


def parse_chariot_object_detections(
    detections: dict,
    image: Image,
    label_key: str = "class",
) -> Prediction:

    if not isinstance(detections, dict):
        raise RuntimeError

    if not isinstance(image, Image):
        raise RuntimeError

    assert len(detections) != 1, "length mismatch"

    # validate
    expected_keys = {
        "num_detections",
        "detection_classes",
        "detection_boxes",
        "detection_scores",
    }
    if set(detections.keys()) != expected_keys:
        raise ValueError(
            f"Expected `dets` to have keys {expected_keys} but got {detections.keys()}"
        )

    # create prediction
    return Prediction(
        datum=image.to_datum(),
        annotations=[
            ScoredAnnotation(
                task_type=enums.TaskType.DETECTION,
                scored_labels=[
                    ScoredLabel(
                        label=Label(key=label_key, value=label),
                        score=float(score),
                    )
                ],
                bounding_box=BoundingBox.from_extrema(
                    ymin=box[0], xmin=box[1], ymax=box[2], xmax=box[3]
                ),
            )
            for box, score, label in zip(
                detections["detection_boxes"],
                detections["detection_scores"],
                detections["detection_classes"],
            )
        ],
    )


def create_model_from_chariot(
    client: Client,
    model: ChariotModel,
    name: str = None,
    description: str = None,
) -> Model:
    """Converts chariot model to a velour model.

    Parameters
    ----------
    client
        Velour client object
    model
        Chariot model object
    name
        (OPTIONAL) Defaults to Chariot model name.
    description
        (OPTIONAL) Defaults to Chariot model description.

    Returns
    -------
    Velour image model
    """

    cv_tasks = [
        ChariotTaskType.IMAGE_AUTOENCODER,
        ChariotTaskType.IMAGE_CLASSIFICATION,
        ChariotTaskType.IMAGE_EMBEDDING,
        ChariotTaskType.IMAGE_GENERATION,
        ChariotTaskType.IMAGE_SEGMENTATION,
        ChariotTaskType.OBJECT_DETECTION,
        ChariotTaskType.OTHER_COMPUTER_VISION,
    ]

    tabular_tasks = [
        ChariotTaskType.STRUCTURED_DATA_CLASSIFICATION,
        ChariotTaskType.STRUCTURED_DATA_REGRESSION,
        ChariotTaskType.OTHER_STRUCTURED_DATA,
    ]

    nlp_tasks = [
        ChariotTaskType.TEXT_CLASSIFICATION,
        ChariotTaskType.TEXT_FILL_MASK,
        ChariotTaskType.TEXT_GENERATION,
        ChariotTaskType.TOKEN_CLASSIFICATION,
        ChariotTaskType.TRANSLATION,
        ChariotTaskType.OTHER_NATURAL_LANGUAGE,
    ]

    if name is None:
        name = f"chariot-{model.name}-v{model.version}"

    if description is None:
        description = model._meta.summary

    href = _construct_url(project_id=model.project_id, model_id=model.id)

    if model.task in cv_tasks:
        pass
    elif model.task in tabular_tasks:
        pass
    if model.task in nlp_tasks:
        raise NotImplementedError(
            f"NLP tasks are currently not supported. '{model.task}'"
        )

    model = Model.create(
        client,
        name=model.id,
        href=href,
        description=description,
    )
    model.add_metadatum(
        MetaDatum(
            key="project_id",
            value=model.project_id,
        )
    )
    model.add_metadatum(
        MetaDatum(
            key="task",
            value=model.task,
        )
    )

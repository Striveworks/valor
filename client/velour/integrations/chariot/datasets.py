import gzip
import json
import tempfile

import requests
from tqdm import tqdm

from velour import enums
from velour.client import Client, ClientException, Dataset
from velour.schemas import (
    Annotation,
    BasicPolygon,
    BoundingBox,
    Datum,
    GroundTruth,
    Label,
    Metadatum,
    Point,
    Polygon,
)


def _retrieve_dataset_version(
    dataset,
    dataset_version_id: str,
):
    if len(dataset.versions) < 1:
        raise ValueError("Chariot dataset has no existing versions.")

    # Get dataset version
    dataset_version = None
    if dataset_version_id is None:
        dataset_version = dataset.versions[-1]  # Use the latest version
    else:
        for version in dataset.versions:
            if version.id == dataset_version_id:
                dataset_version = version
                break
        if dataset_version is None:
            raise ValueError(
                f"Chariot DatasetVersion not found with id: {dataset_version_id}"
            )

    return dataset_version


def _retrieve_dataset_manifest(
    manifest_url: str,
    disable_progress_bar: bool = False,
):
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
            desc="Downloading manifest",
            disable=disable_progress_bar,
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
            if line is None:
                raise ValueError(
                    f"manifest url `{manifest_url}` returned null."
                )
            elif line == "":
                raise ValueError(
                    f"manifest url `{manifest_url}` returned empty string."
                )
            chariot_dataset.append(json.loads(line))

    return chariot_dataset


def _parse_labels(annotation: dict):
    labels = []
    if "class_label" in annotation:
        labels = [Label(key="class_label", value=annotation["class_label"])]
    labels.extend(
        [
            Label(key=attribute, value=annotation["attributes"][attribute])
            for attribute in annotation["attributes"]
        ]
    )
    return labels


def _parse_annotation(dataset_version, annotation: dict):
    task_types = []
    labels = _parse_labels(annotation)
    bounding_box = None
    polygon = None
    multipolygon = None
    raster = None

    # Image Classification
    if dataset_version.supported_task_types.image_classification:
        task_types.append(enums.TaskType.CLF)

    # Image Object Detection
    if dataset_version.supported_task_types.object_detection:
        task_types.append(enums.TaskType.DET)
        if "bbox" in annotation:
            bounding_box = BoundingBox.from_extrema(
                xmin=annotation["bbox"]["xmin"],
                ymin=annotation["bbox"]["ymin"],
                xmax=annotation["bbox"]["xmax"],
                ymax=annotation["bbox"]["ymax"],
            )

    # Image Segmentation
    if dataset_version.supported_task_types.image_segmentation:
        task_types.append(enums.TaskType.SEG)
        if "contours" in annotation:
            polygon = Polygon(
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
            )

    # Text Sentiment
    if dataset_version.supported_task_types.text_sentiment:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    # Text Summarization
    if dataset_version.supported_task_types.text_summarization:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    # Text Token Classifier
    if dataset_version.supported_task_types.text_token_classification:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    # Text Translation
    if dataset_version.supported_task_types.text_translation:
        raise NotImplementedError(
            "Text-based datasets not currently supported."
        )

    return Annotation(
        task_type=task_types[-1],  # @TODO Make this better.
        labels=labels,
        bounding_box=bounding_box,
        polygon=polygon,
        multipolygon=multipolygon,
        raster=raster,
    )


def _parse_groundtruth_from_evaluation_manifest(
    dataset_version, manifest_datum: dict
):
    return GroundTruth(
        datum=Datum(
            uid=manifest_datum["datum_id"],
            metadata=[Metadatum(key="path", value=manifest_datum["path"])],
        ),
        annotations=[
            _parse_annotation(dataset_version, annotation)
            for annotation in manifest_datum["annotations"]
        ],
    )


def create_dataset_from_chariot(
    client: Client, dataset, dataset_version_id: str
):
    """Create empty dataset from Chariot dataset attributes.

    Creates a new Velour dataset from a Chariot dataset.

    Parameters
    ----------
    client: velour.client.Client
    dataset: chariot.datasets.Dataset
    dataset_version_id: str
        Chariot dataset version id.

    Returns
    ----------
    velour.client.Dataset
        Velour dataset object links to the dataset on the backend.
        The dataset will be empty and ready for groundtruths.

    Raises
    ----------
    velour.client.ClientException
        Any error propagated from the backend.
    """
    return Dataset.create(
        client=client,
        name=dataset_version_id,
        integration="chariot",
        title=dataset.name,
        description=dataset._meta.description,
        project_id=dataset.project_id,
        dataset_id=dataset.id,
    )


def get_groundtruth_parser_from_chariot(dataset_version):
    """Returns groundtruth parser with dataset_version object embedded."""

    def velour_parser(annotation: dict):
        return _parse_annotation(dataset_version, annotation)

    return velour_parser


def get_chariot_dataset_integration(
    client: Client,
    dataset,
    dataset_version_id: str = None,
):
    """Returns Velour dataset and groundtruth parser.

    Creates or gets a new Velour dataset from a Chariot dataset. Returns
    Velour dataset along with matched Chariot groundtruth parser.

    Parameters
    ----------
    client: velour.client.Client
    dataset: chariot.datasets.Dataset
    dataset_version_id: str
        Chariot dataset version id.

    Returns
    ----------
    tuple
        dataset: velour.client.Dataset
        parser: callable(annotation: dict)
            Parsing function that converts Chariot groundtruth annotations.

    Raises
    ----------
    velour.client.ClientException
        Any error propagated from the backend.
    """

    # get chariot dataset version
    dataset_version = _retrieve_dataset_version(dataset, dataset_version_id)

    # create or get dataset
    try:
        velour_dataset = Dataset.get(client, dataset_version.id)
    except ClientException as e:
        if "does not exist" not in str(e):
            raise e
        velour_dataset = create_dataset_from_chariot(
            client, dataset, dataset_version_id
        )

    # get parsing function
    velour_parser = get_groundtruth_parser_from_chariot(dataset_version)

    return (velour_dataset, velour_parser)


def create_dataset_from_chariot_evaluation_manifest(
    client: Client,
    dataset,
    dataset_version_id,
    disable_progress_bar: bool = False,
) -> Dataset:
    """Create dataset from an evaluation manifest.

    Creates a finalized Velour dataset from an existing Chariot dataset version. Note that
    this does not expose image metadata and therefore metadata-based filtering
    is not supported.

    Parameters
    ----------
    client: velour.client.Client
    dataset: chariot.datasets.Dataset
    dataset_version_id: str
        Chariot dataset version id.
    disable_progress_bar: bool
        Toggles tqdm progress bar.

    Returns
    ----------
    velour.client.Dataset
        Velour dataset object that links to the dataset on the backend.
        The dataset will be finalized and ready for evaluation jobs.

    Raises
    ----------
    velour.client.ClientException
        Any error propagated from the backend.
    """
    # get chariot dataset version
    dataset_version = _retrieve_dataset_version(dataset, dataset_version_id)

    # Create new dataset
    velour_dataset = Dataset.create(
        client=client,
        name=dataset_version.id,
        integration="chariot",
        title=dataset.name,
        description=dataset._meta.description,
        project_id=dataset.project_id,
        dataset_id=dataset.id,
    )

    # Retrieve the manifest url
    manifest_url = dataset_version.get_evaluation_manifest_url()

    # Retrieve the manifest
    manifest = _retrieve_dataset_manifest(manifest_url, disable_progress_bar)

    # Create velour groundtruths
    for datum in tqdm(
        manifest,
        desc="Uploading to Velour",
        unit="datum",
        disable=disable_progress_bar,
    ):
        gt = _parse_groundtruth_from_evaluation_manifest(
            dataset_version, datum
        )
        velour_dataset.add_groundtruth(gt)

    # Finalize groundtruths
    velour_dataset.finalize()

    return velour_dataset

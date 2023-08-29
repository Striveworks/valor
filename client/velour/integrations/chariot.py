import gzip
import json
import tempfile
from typing import Tuple

import requests
from tqdm import tqdm

from velour import enums
from velour.client import Client, ClientException, Dataset, Model
from velour.schemas import (
    Annotation,
    BasicPolygon,
    BoundingBox,
    Datum,
    GroundTruth,
    Label,
    MetaDatum,
    Point,
    Polygon,
    Prediction,
    ScoredAnnotation,
    ScoredLabel,
)

""" Dataset """


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


def _parse_groundtruth(dataset_version, manifest_datum: dict):
    def _parse_annotation(dataset_version, annotation: dict):
        task_types = []
        labels = []
        if "class_label" in annotation:
            labels += [
                Label(key="class_label", value=annotation["class_label"])
            ]
        labels += [
            Label(key=attribute, value=annotation["attributes"][attribute])
            for attribute in annotation["attributes"]
        ]
        bounding_box = None
        polygon = None
        multipolygon = None
        raster = None

        # Image Classification
        if dataset_version.supported_task_types.image_classification:
            task_types.append(enums.TaskType.CLASSIFICATION)

        # Image Object Detection
        if dataset_version.supported_task_types.object_detection:
            task_types.append(enums.TaskType.DETECTION)
            if "bbox" in annotation:
                bounding_box = BoundingBox.from_extrema(
                    xmin=annotation["bbox"]["xmin"],
                    ymin=annotation["bbox"]["ymin"],
                    xmax=annotation["bbox"]["xmax"],
                    ymax=annotation["bbox"]["ymax"],
                )

        # Image Segmentation
        if dataset_version.supported_task_types.image_segmentation:
            task_types.append(enums.TaskType.SEMANTIC_SEGMENTATION)
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

    return GroundTruth(
        datum=Datum(
            uid=manifest_datum["datum_id"],
            metadata=[MetaDatum(key="path", value=manifest_datum["path"])],
        ),
        annotations=[
            _parse_annotation(dataset_version, annotation)
            for annotation in manifest_datum["annotations"]
        ],
    )


def get_chariot_dataset_integration(
    client: Client,
    dataset,
    dataset_version_id: str = None,
    disable_progress_bar: bool = False,
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

    # Check if dataset has already been created
    try:
        velour_dataset = Dataset.get(client, dataset_version.id)
    except ClientException as e:
        if "does not exist" not in str(e):
            raise e

        velour_dataset = Dataset.create(
            client=client,
            name=dataset_version.id,
            integration="chariot",
            title=dataset.name,
            description=dataset._meta.description,
            project_id=dataset.project_id,
            dataset_id=dataset.id,
        )

        # Retrieve the manifest
        manifest_url = dataset_version.get_evaluation_manifest_url()
        manifest = _retrieve_dataset_manifest(
            manifest_url, disable_progress_bar
        )

        # Create velour groundtruths
        for datum in tqdm(
            manifest,
            desc="Uploading to Velour",
            unit="datum",
            disable=disable_progress_bar,
        ):
            gt = _parse_groundtruth(dataset_version, datum)
            velour_dataset.add_groundtruth(gt)

        # Finalize groundtruths
        velour_dataset.finalize()

    # generate parsing function
    def velour_parser(manifest_datum):
        return _parse_groundtruth(dataset_version, manifest_datum)

    return (velour_dataset, velour_parser)


""" Model """


def _parse_chariot_predict_proba_image_classification(
    datum: Datum,
    labels: dict,
    result: list,
    label_key: str = "class_label",
):
    # validate result
    if not isinstance(result, list):
        raise TypeError(
            "image classification epected format List[List[float]]"
        )
    if len(result) != 1:
        raise ValueError("cannot have more than one result per datum")
    if not isinstance(result[0], list):
        raise TypeError(
            "image classification epected format List[List[float]]"
        )
    if len(labels) != len(result[0]):
        raise ValueError("number of labels does not equal number of scores")

    # create prediction
    labels = {v: k for k, v in labels.items()}
    return Prediction(
        datum=datum,
        annotations=[
            ScoredAnnotation(
                task_type=enums.TaskType.CLASSIFICATION,
                scored_labels=[
                    ScoredLabel(
                        label=Label(key=label_key, value=labels[i]),
                        score=score,
                    )
                    for i, score in enumerate(result[0])
                ],
            )
        ],
    )


def _parse_chariot_predict_image_classification(
    datum: Datum,
    labels: dict,
    result: list,
    label_key: str = "class_label",
):
    # validate result
    if not isinstance(result, list):
        raise TypeError(
            "image classification epected format List[List[float]]"
        )
    if len(result) != 1:
        raise ValueError("cannot have more than one result per datum")
    if not isinstance(result[0], list):
        raise TypeError(
            "image classification epected format List[List[float]]"
        )
    if len(labels) != len(result[0]):
        raise ValueError("number of labels does not equal number of scores")

    # create prediction
    labels = {v: k for k, v in labels.items()}
    return Prediction(
        datum=datum,
        annotations=[
            ScoredAnnotation(
                task_type=enums.TaskType.CLASSIFICATION,
                scored_labels=[
                    ScoredLabel(
                        label=Label(key=label_key, value=label),
                        score=1.0 if label == result[0] else 0.0,
                    )
                    for label in labels
                ],
            )
        ],
    )


def _parse_chariot_image_object_detection_with_action_detect(
    datum: Datum,
    result: dict,
    label_key: str = "class_label",
):
    # validate result
    if not isinstance(result, list):
        raise TypeError("image object detection epected format List[Dict[]]")
    if len(result) != 1:
        raise ValueError("cannot have more than one result per datum")
    if not isinstance(result[0], dict):
        raise TypeError("image object detection epected format List[Dict[]]")
    result = result[0]

    # validate result
    expected_keys = {
        "num_detections",
        "detection_classes",
        "detection_boxes",
        "detection_scores",
    }
    if set(result.keys()) != expected_keys:
        raise ValueError(
            f"Expected `dets` to have keys {expected_keys} but got {result.keys()}"
        )

    # create prediction
    return Prediction(
        datum=datum,
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
                result["detection_boxes"],
                result["detection_scores"],
                result["detection_classes"],
            )
        ],
    )


def get_chariot_inference_model(
    client: Client,
    model,
):
    try:
        return Model.get(client, model.id)
    except ClientException as e:
        if "does not exist" not in str(e):
            raise e
        return Model.create(
            client=client,
            name=model.id,
            integration="chariot",
            title=model.name,
            description=model._meta.summary,
            project_id=model.project_id,
        )


def get_chariot_inference_parser(
    task_type: str, action: str, label_key: str, class_labels: list = None
):

    if task_type == "Image Classification":

        if action == "predict":

            def velour_parser(datum: Datum, result):
                return _parse_chariot_predict_image_classification(
                    datum, class_labels, result=result, label_key=label_key
                )

        elif action == "predict_proba":

            def velour_parser(datum: Datum, result):
                return _parse_chariot_predict_proba_image_classification(
                    datum, class_labels, result=result, label_key=label_key
                )

        else:
            raise NotImplementedError(
                f"action `{action}` not supported for task type `Image Classification`"
            )

    elif task_type == "Object Detection":

        def velour_parser(datum: Datum, result):
            return _parse_chariot_image_object_detection_with_action_detect(
                datum=datum,
                result=result,
                label_key=label_key,
            )

    elif task_type == "Image Segmentation":
        raise NotImplementedError(
            "Image Segmentation has not been implemented."
        )
    else:
        raise NotImplementedError(task_type)

    return velour_parser


def get_chariot_model_integration(
    client: Client, model, action: str, label_key: str = "class_label"
) -> Tuple[Model, callable]:
    """Returns tuple of (velour.client.Model, parsing_fn(datum, result))"""

    # check if model has already been created
    velour_model = get_chariot_inference_model(client, model)

    # retrieve task-related parser
    velour_parser = get_chariot_inference_parser(
        task_type=model.task.value,
        action=action,
        label_key=label_key,
        class_labels=model.class_labels,
    )

    return (velour_model, velour_parser)

import gzip
import json
import tempfile
import urllib

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

try:
    import chariot
    from chariot.datasets import Dataset as ChariotDataset
    from chariot.models import Model as ChariotModel
except ModuleNotFoundError:
    "`chariot` package not found. if you have an account on Chariot please see https://production.chariot.striveworks.us/docs/sdk/sdk for how to install the python SDK"


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


class ChariotDatasetIntegration:
    def __init__(
        self,
        client: Client,
        dataset: ChariotDataset,
        dataset_version_id: str = None,
        back_populate: bool = False,  # @TODO: Option to have velour download the images to scrape sizing
    ):
        if len(dataset.versions) < 1:
            raise ValueError("Chariot dataset has no existing versions.")

        self.client = client
        self._dataset = dataset

        # Get dataset version
        self.dsv = None
        if dataset_version_id is None:
            self.dsv = dataset.versions[-1]  # Use the latest version
        else:
            for version in dataset.versions:
                if version.id == dataset_version_id:
                    self.dsv = version
                    break
            if self.dsv is None:
                raise ValueError(
                    f"Chariot DatasetVersion not found with id: {dataset_version_id}"
                )

        # Retrieve the manifest
        manifest_url = self.dsv.get_evaluation_manifest_url()
        manifest = self._retrieve_dataset_manifest(manifest_url)

        # Generate set of datum id
        self.datum_info = {
            datum["datum_id"]: {
                "path": datum["path"],
                "annotations": [
                    self._parse_annotation(annotation)
                    for annotation in datum["annotations"]
                ],
            }
            for datum in manifest
        }

    def _retrieve_dataset_manifest(self, manifest_url: str):
        """Retrieves and unpacks Chariot dataset annotations from a manifest url."""

        chariot_dataset = []

        # Create a temporary file
        with tempfile.TemporaryFile(mode="w+b") as f:
            # Download compressed jsonl file
            response = requests.get(manifest_url, stream=True)

            # Progress bar
            total_size_in_bytes = int(
                response.headers.get("content-length", 0)
            )
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                desc="Downloading manifest",
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

    def _parse_annotation(self, annotation):

        task_types = []
        labels = [
            Label(key="class_label", value=annotation["class_label"])
        ] + [
            Label(key=attribute, value=annotation["attributes"][attribute])
            for attribute in annotation["attributes"]
        ]
        bounding_box = None
        polygon = None
        multipolygon = None
        raster = None

        # Image Classification
        if self.dsv.supported_task_types.image_classification:
            task_types.append(enums.TaskType.CLASSIFICATION)

        # Image Object Detection
        if self.dsv.supported_task_types.object_detection:
            task_types.append(enums.TaskType.DETECTION)
            bounding_box = BoundingBox.from_extrema(
                xmin=annotation["bbox"]["xmin"],
                ymin=annotation["bbox"]["ymin"],
                xmax=annotation["bbox"]["xmax"],
                ymax=annotation["bbox"]["ymax"],
            )

        # Image Segmentation
        if self.dsv.supported_task_types.image_segmentation:
            task_types.append(enums.TaskType.SEMANTIC_SEGMENTATION)
            polygon = (
                Polygon(
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

        # Text Sentiment
        if self.dsv.supported_task_types.text_sentiment:
            raise NotImplementedError(
                "Text-based datasets not currently supported."
            )

        # Text Summarization
        if self.dsv.supported_task_types.text_summarization:
            raise NotImplementedError(
                "Text-based datasets not currently supported."
            )

        # Text Token Classifier
        if self.dsv.supported_task_types.text_token_classification:
            raise NotImplementedError(
                "Text-based datasets not currently supported."
            )

        # Text Translation
        if self.dsv.supported_task_types.text_translation:
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

    @property
    def dataset(self) -> Dataset:
        try:
            return Dataset.get(self.client, self._dataset.id)
        except ClientException as e:
            if "does not exist" not in str(e):
                raise e
            return Dataset.create(
                client=self.client,
                name=self._dataset.id,
                integration="chariot",
                title=self._dataset.name,
                description=self._dataset._meta.description,
                project_id=self._dataset.project_id,
                dataset_version_id=self.dsv.id,
                type=self.dsv.type,
            )

    def add_datum(self, datum: Datum):
        if datum.uid not in self.datum_info:
            raise KeyError(
                f"datum {datum.uid} does not exist in annotation set"
            )

        # add datum path to metadata
        datum.metadata.append(
            MetaDatum(
                key="path",
                value=self.datum_info[datum.uid]["path"],
            )
        )

        # construct groundtruth
        gt = GroundTruth(
            datum=datum,
            annotations=self.datum_info[datum.uid]["annotations"],
        )

        # add groundtruth to dataset
        self.dataset.add_groundtruth(gt)


class ChariotModelIntegration:
    def __init__(
        self,
        client: Client,
        model: ChariotModel,
    ):
        self.client = client
        self._model = model

    def _parse_chariot_annotation(self, annotation):
        pass

    @property
    def model(self) -> Model:
        try:
            return Model.get(self.client, self._model.id)
        except ClientException as e:
            if "does not exist" not in str(e):
                raise e
            return Model.create(
                client=self.client,
                name=self._model.id,
                integration="chariot",
                title=self._model.name,
                description=self._model._meta.summary,
                project_id=self._model.project_id,
            )

    def add_prediction(
        self,
        datum: Datum,
        action: str,
        result: dict,
        label_key: str = "class_label",
    ):

        if action == "detect":
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
            pd = Prediction(
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

            self.model.add_prediction(pd)

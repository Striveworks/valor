import json
import math
import os
import time
import warnings
from dataclasses import asdict
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import requests

from velour import enums, schemas
from velour.coretypes import Datum, GroundTruth, Label, Prediction
from velour.enums import JobStatus, State
from velour.filters import BinaryExpression, DeclarativeMapper, create_filter
from velour.metatypes import ImageMetadata
from velour.schemas.metadata import validate_metadata


class ClientException(Exception):
    pass


class Client:
    """Client for interacting with the velour backend"""

    def __init__(self, host: str, access_token: str = None):
        """
        Parameters
        ----------
        host
            the host to connect to. Should start with "http://" or "https://"
        access_token
            the access token if the host requires authentication
        """
        if not (host.startswith("http://") or host.startswith("https://")):
            raise ValueError(
                f"host must stat with 'http://' or 'https://' but got {host}"
            )

        if not host.endswith("/"):
            host += "/"
        self.host = host
        self.access_token = os.getenv("VELOUR_ACCESS_TOKEN", access_token)

        # check the connection by hitting the users endpoint
        email = self._get_users_email()
        success_str = f"Succesfully connected to {self.host}"
        success_str += f" with user {email}." if email else "."
        print(success_str)

    def _get_users_email(
        self,
    ) -> Union[str, None]:
        """Gets the users e-mail address (in the case when auth is enabled)
        or returns None in the case of a no-auth backend.
        """
        resp = self._requests_get_rel_host("user").json()
        return resp["email"]

    def _requests_wrapper(
        self, method_name: str, endpoint: str, *args, **kwargs
    ):
        assert method_name in ["get", "post", "put", "delete"]

        if endpoint[0] == "/":
            raise ValueError(
                "`endpoint` should not start with a forward slash."
            )

        url = urljoin(self.host, endpoint)
        requests_method = getattr(requests, method_name)

        if self.access_token is not None:
            headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            headers = None
        resp = requests_method(url, headers=headers, *args, **kwargs)
        if not resp.ok:
            try:
                raise ClientException(resp.json()["detail"])
            except (requests.exceptions.JSONDecodeError, KeyError):
                resp.raise_for_status()

        return resp

    def _requests_post_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="post", endpoint=endpoint, *args, **kwargs
        )

    def _requests_get_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="get", endpoint=endpoint, *args, **kwargs
        )

    def _requests_put_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="put", endpoint=endpoint, *args, **kwargs
        )

    def _requests_delete_rel_host(self, endpoint: str, *args, **kwargs):
        return self._requests_wrapper(
            method_name="delete", endpoint=endpoint, *args, **kwargs
        )

    def get_bulk_evaluations(
        self,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
    ) -> List[dict]:
        """
        Returns all metrics associated with user-supplied dataset and model names

        Parameters
        ----------
        models
            A list of dataset names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.
        datasets
            A list of model names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.
        """

        if not (models or datasets):
            raise ValueError(
                "Please provide atleast one model name or dataset name"
            )

        if models:
            # let users just pass one name as a string
            if isinstance(models, str):
                models = [models]
            model_params = ",".join(models)
        else:
            model_params = None

        if datasets:
            if isinstance(datasets, str):
                datasets = [datasets]
            dataset_params = ",".join(datasets)
        else:
            dataset_params = None

        if model_params and dataset_params:
            endpoint = (
                f"evaluations?models={model_params}&datasets={dataset_params}"
            )
        elif model_params:
            endpoint = f"evaluations?models={model_params}"
        else:
            endpoint = f"evaluations?datasets={dataset_params}"

        evals = self._requests_get_rel_host(endpoint).json()
        return evals

    def get_datasets(
        self,
    ) -> List[dict]:
        return self._requests_get_rel_host("datasets").json()

    def get_models(
        self,
    ) -> List[dict]:
        return self._requests_get_rel_host("models").json()

    def get_labels(
        self,
    ) -> List[Label]:
        return self._requests_get_rel_host("labels").json()

    def delete_dataset(self, name: str, timeout: int = 0) -> None:
        """
        Delete a dataset using FastAPI's BackgroundProcess

        Parameters
        ----------
        name
            The name of the dataset to be deleted
        timeout
            The number of seconds to wait in order to confirm that the dataset was deleted
        """
        try:
            self._requests_delete_rel_host(f"datasets/{name}")

            if timeout:
                for _ in range(timeout):
                    if self.get_dataset_status(name) == State.NONE:
                        break
                    else:
                        time.sleep(1)
                else:
                    raise TimeoutError(
                        "Dataset wasn't deleted within timeout interval"
                    )

        except ClientException as e:
            if "does not exist" not in str(e):
                raise e

    def delete_model(self, name: str, timeout: int = 0) -> None:
        """
        Delete a model using FastAPI's BackgroundProcess

        Parameters
        ----------
        name
            The name of the model to be deleted
        timeout
            The number of seconds to wait in order to confirm that the model was deleted
        """
        try:
            self._requests_delete_rel_host(f"models/{name}")

            if timeout:
                for _ in range(timeout):
                    if self.get_dataset_status(name) == State.NONE:
                        break
                    else:
                        time.sleep(1)
                else:
                    raise TimeoutError(
                        "Model wasn't deleted within timeout interval"
                    )

        except ClientException as e:
            if "does not exist" not in str(e):
                raise e

    def get_dataset_status(
        self,
        dataset_name: str,
    ) -> State:
        try:
            resp = self._requests_get_rel_host(
                f"datasets/{dataset_name}/status"
            ).json()
        except Exception:
            resp = State.NONE

        return resp


class Evaluation:
    def __init__(
        self,
        client: Client,
        job_id: int,
        dataset: str,
        model: str,
        **kwargs,
    ):
        self._id: int = job_id
        self._client: Client = client
        self.dataset = dataset
        self.model = model

        settings = self._client._requests_get_rel_host(
            f"evaluations/{self._id}/settings"
        ).json()
        self._settings = schemas.EvaluationJob(**settings)

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def id(
        self,
    ) -> int:
        return self._id

    @property
    def settings(
        self,
    ) -> schemas.EvaluationJob:
        return self._settings

    @property
    def status(
        self,
    ) -> str:
        resp = self._client._requests_get_rel_host(
            f"evaluations/{self._id}"
        ).json()
        return JobStatus(resp)

    @property
    def type(
        self,
    ) -> enums.TaskType:
        return self._settings.type

    @property
    def task_type(
        self,
    ) -> enums.TaskType:
        return self._settings.task_type

    @property
    def annotation_type(
        self,
    ) -> enums.TaskType:
        return self._settings.annotation_type

    @property
    def parameters(
        self,
    ) -> dict:
        return self._settings.parameters

    @property
    def metrics(
        self,
    ) -> List[dict]:
        if self.status != JobStatus.DONE:
            return []
        return self._client._requests_get_rel_host(
            f"evaluations/{self._id}/metrics"
        ).json()

    @property
    def confusion_matrices(
        self,
    ) -> List[dict]:
        if self.status != JobStatus.DONE:
            return []
        return self._client._requests_get_rel_host(
            f"evaluations/{self._id}/confusion-matrices"
        ).json()

    def wait_for_completion(self, *, interval=1.0, timeout=None):
        if timeout:
            timeout_counter = int(math.ceil(timeout / interval))
        while self.status not in [JobStatus.DONE, JobStatus.FAILED]:
            time.sleep(interval)
            if timeout:
                timeout_counter -= 1
                if timeout_counter < 0:
                    raise TimeoutError


class Dataset:
    id = DeclarativeMapper("dataset.id", int)
    name = DeclarativeMapper("dataset.name", str)
    metadata = DeclarativeMapper("dataset.metadata", Union[int, float, str])

    def __init__(self):
        self.client: Client = None
        self.id: int = None
        self.name: str = None
        self.metadata: dict = None

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        metadata: Dict[str, Union[int, float, str, schemas.GeoJSON]] = None,
        id: Union[int, None] = None,
    ):
        dataset = cls()
        dataset.client = client
        dataset.name = name
        dataset.metadata = metadata
        dataset.id = id
        dataset._validate()
        client._requests_post_rel_host("datasets", json=dataset.dict())
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        resp = client._requests_get_rel_host(f"datasets/{name}").json()
        dataset = cls()
        dataset.client = client
        dataset.name = resp["name"]
        dataset.metadata = resp["metadata"]
        dataset.id = resp["id"]
        dataset._validate()
        return dataset

    def _validate(self):
        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("`id` should be of type `int`")
        if not self.metadata:
            self.metadata = {}
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
        }

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            raise TypeError(f"Expected type `{type(Dataset)}`, got `{other}`")
        return self.dict() == other.dict()

    def add_groundtruth(
        self,
        groundtruth: GroundTruth,
    ):
        try:
            assert isinstance(groundtruth, GroundTruth)
        except AssertionError:
            raise TypeError(f"Invalid type `{type(groundtruth)}`")

        if len(groundtruth.annotations) == 0:
            warnings.warn(
                f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations. Skipping..."
            )
            return

        groundtruth.datum.dataset = self.name
        self.client._requests_post_rel_host(
            "groundtruths",
            json=groundtruth.dict(),
        )

    def get_groundtruth(self, uid: str) -> GroundTruth:
        resp = self.client._requests_get_rel_host(
            f"groundtruths/dataset/{self.name}/datum/{uid}"
        ).json()
        return GroundTruth(**resp)

    def get_labels(
        self,
    ) -> List[Label]:
        labels = self.client._requests_get_rel_host(
            f"labels/dataset/{self.name}"
        ).json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def get_datums(
        self,
    ) -> List[Datum]:
        """Returns a list of datums."""
        datums = self.client._requests_get_rel_host(
            f"data/dataset/{self.name}"
        ).json()
        return [Datum(**datum) for datum in datums]

    def get_images(
        self,
    ) -> List[ImageMetadata]:
        """Returns a list of Image Metadata if it exists, otherwise raises Dataset contains no images."""
        return [
            ImageMetadata.from_datum(datum)
            for datum in self.get_datums()
            if ImageMetadata.valid(datum)
        ]

    def get_evaluations(
        self,
    ) -> List[Evaluation]:
        model_evaluations = self.client._requests_get_rel_host(
            f"evaluations/datasets/{self.name}"
        ).json()
        return [
            Evaluation(
                client=self.client,
                dataset=self.name,
                model=model_name,
                job_id=job_id,
            )
            for model_name in model_evaluations
            for job_id in model_evaluations[model_name]
        ]

    def finalize(
        self,
    ):
        return self.client._requests_put_rel_host(
            f"datasets/{self.name}/finalize"
        )

    def delete(
        self,
    ):
        self.client._requests_delete_rel_host(f"datasets/{self.name}").json()
        del self


class Model:
    id = DeclarativeMapper("model.id", int)
    name = DeclarativeMapper("model.name", str)
    metadata = DeclarativeMapper("model.metadata", Union[int, float, str])

    def __init__(self):
        self.client: Client = None
        self.id: int = None
        self.name: str = ""
        self.metadata: dict = None

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        metadata: Dict[str, Union[int, float, str, schemas.GeoJSON]] = None,
        id: Union[int, None] = None,
    ):
        model = cls()
        model.client = client
        model.name = name
        model.metadata = metadata
        model.id = id
        model._validate()
        client._requests_post_rel_host("models", json=model.dict())
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        resp = client._requests_get_rel_host(f"models/{name}").json()
        model = cls()
        model.client = client
        model.name = resp["name"]
        model.metadata = resp["metadata"]
        model.id = resp["id"]
        model._validate()
        return model

    def _validate(self):
        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("`id` should be of type `int`")
        if not self.metadata:
            self.metadata = {}
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
        }

    def __eq__(self, other):
        if not isinstance(other, Model):
            raise TypeError(f"Expected type `{type(Model)}`, got `{other}`")
        return self.dict() == other.dict()

    def add_prediction(self, prediction: Prediction):
        try:
            assert isinstance(prediction, Prediction)
        except AssertionError:
            raise TypeError(
                f"Expected `velour.Prediction`, got `{type(prediction)}`"
            )

        if len(prediction.annotations) == 0:
            warnings.warn(
                f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations. Skipping..."
            )
            return

        prediction.model = self.name
        return self.client._requests_post_rel_host(
            "predictions",
            json=prediction.dict(),
        )

    def finalize_inferences(self, dataset: "Dataset") -> None:
        return self.client._requests_put_rel_host(
            f"models/{self.name}/datasets/{dataset.name}/finalize"
        ).json()

    def evaluate_classification(
        self, dataset: Dataset, timeout: Optional[int] = None
    ) -> Evaluation:
        """Start a classification evaluation job

        Parameters
        ----------
        dataset
            The dataset to evaluate against
        group_by
            Optional name of metadatum to group the results by
        timeout
            The number of seconds to wait for the job to finish. Used to ensure deterministic behavior when testing.

        Returns
        -------
        Evaluation
            a job object that can be used to track the status of the job
            and get the metrics of it upon completion
        """

        evaluation = schemas.EvaluationJob(
            model=self.name,
            dataset=dataset.name,
        )

        resp = self.client._requests_post_rel_host(
            "evaluations/clf-metrics", json=asdict(evaluation)
        ).json()

        evaluation_job = Evaluation(
            client=self.client,
            dataset=dataset.name,
            model=self.name,
            **resp,
        )

        # blocking behavior
        if timeout:
            evaluation_job.wait_for_completion(interval=1.0, timeout=timeout)

        return evaluation_job

    def evaluate_detection(
        self,
        dataset: "Dataset",
        iou_thresholds_to_compute: List[float] = None,
        iou_thresholds_to_keep: List[float] = None,
        filters: Union[Dict, List[BinaryExpression]] = None,
        timeout: Optional[int] = None,
    ) -> Evaluation:
        """Evaluate object detections."""

        # Default iou thresholds
        if iou_thresholds_to_compute is None:
            iou_thresholds_to_compute = [
                round(0.5 + 0.05 * i, 2) for i in range(10)
            ]
        if iou_thresholds_to_keep is None:
            iou_thresholds_to_keep = [0.5, 0.75]

        parameters = schemas.DetectionParameters(
            iou_thresholds_to_compute=iou_thresholds_to_compute,
            iou_thresholds_to_keep=iou_thresholds_to_keep,
        )

        if not isinstance(filters, dict):
            filters = create_filter(filters)

        evaluation = schemas.EvaluationJob(
            model=self.name,
            dataset=dataset.name,
            settings=schemas.EvaluationSettings(
                parameters=parameters,
                filters=filters,
            ),
        )

        resp = self.client._requests_post_rel_host(
            "evaluations/ap-metrics", json=asdict(evaluation)
        ).json()

        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects

        for k in ["missing_pred_labels", "ignored_pred_labels"]:
            resp[k] = [Label(**la) for la in resp[k]]

        evaluation_job = Evaluation(
            client=self.client,
            dataset=dataset.name,
            model=self.name,
            **resp,
        )

        # blocking behavior
        if timeout:
            evaluation_job.wait_for_completion(interval=1.0, timeout=timeout)

        return evaluation_job

    def evaluate_segmentation(
        self, dataset: Dataset, timeout: Optional[int] = None
    ) -> Evaluation:
        evaluation = schemas.EvaluationJob(
            model=self.name,
            dataset=dataset.name,
        )

        resp = self.client._requests_post_rel_host(
            "evaluations/semantic-segmentation-metrics",
            json=asdict(evaluation),
        ).json()

        evaluation_job = Evaluation(
            client=self.client,
            dataset=dataset.name,
            model=self.name,
            **resp,
        )

        # blocking behavior
        if timeout:
            evaluation_job.wait_for_completion(interval=1.0, timeout=timeout)

        return evaluation_job

    def delete(
        self,
    ):
        self.client._requests_delete_rel_host(f"models/{self.name}").json()
        del self

    def get_prediction(self, datum: Datum) -> Prediction:
        resp = self.client._requests_get_rel_host(
            f"predictions/model/{self.name}/dataset/{datum.dataset}/datum/{datum.uid}",
        ).json()
        return Prediction(**resp)

    def get_labels(
        self,
    ) -> List[Label]:
        labels = self.client._requests_get_rel_host(
            f"labels/model/{self.name}"
        ).json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def get_evaluations(
        self,
    ) -> List[Evaluation]:
        dataset_evaluations = self.client._requests_get_rel_host(
            f"evaluations/models/{self.name}"
        ).json()
        return [
            Evaluation(
                client=self.client,
                dataset=dataset_name,
                model=self.name,
                job_id=job_id,
            )
            for dataset_name in dataset_evaluations
            for job_id in dataset_evaluations[dataset_name]
        ]

    def get_metric_dataframes(
        self,
    ):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Must have pandas installed to use `get_metric_dataframes`."
            )

        ret = []
        for evaluation in self.get_evaluations():
            metrics = [
                {**metric, "dataset": evaluation.dataset}
                for metric in evaluation.metrics["metrics"]
            ]
            df = pd.DataFrame(metrics)
            for k in ["label", "parameters"]:
                df[k] = df[k].fillna("n/a")
            df["parameters"] = df["parameters"].apply(json.dumps)
            df["label"] = df["label"].apply(
                lambda x: f"{x['key']}: {x['value']}" if x != "n/a" else x
            )
            df = df.pivot(
                index=["type", "parameters", "label"], columns=["dataset"]
            )
            ret.append({"settings": evaluation.settings, "df": df})

        return ret

    # TODO Endpoint is independent of model, should be moved to Client?
    def get_evaluation_status(
        self,
        job_id: int,
    ) -> State:
        resp = self.client._requests_get_rel_host(
            f"evaluations/{job_id}"
        ).json()
        return resp

import json
import math
import os
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests

from velour import enums, schemas
from velour.enums import AnnotationType, JobStatus, State


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

    def get_ranked_evaluations(
        self,
        dataset_name: str,
        metric: str,
        parameters: dict = None,
        label_keys: dict = None,
        rank_from_highest_value_to_lowest_value: bool = True,
    ):
        """
        Returns all metrics associated with a particular dataset, ranked according to user inputs

        Parameters
        ----------
        dataset_name
            The dataset name for which to fetch metrics for.
        metric
            The metric to use when ranking evaluations (e.g., "mAP")
        parameters
            The metric parameters to filter on when computing the ranking (e.g., {'iou':.5}). Will raise a ValueError if the user supplies a metric which requires more granular parameters.
        label_keys
            The list of label keys to filter on (e.g., ['key1'])
        rank_from_highest_value_to_lowest_value
            A boolean to indicate whether the metric values should be ranked from highest to lowest
        """

        endpoint = f"ranked-evaluations?dataset_name={dataset_name}&metric={metric}&parameters={json.dumps(parameters)}&label_keys={json.dumps(label_keys)}&rank_from_highest_value_to_lowest_value={rank_from_highest_value_to_lowest_value}"
        ranked_evals = self._requests_get_rel_host(endpoint).json()
        return ranked_evals

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
    ) -> List[schemas.Label]:
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
    ) -> List[schemas.Metadatum]:
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
    def __init__(
        self,
        client: Client,
        info: schemas.Dataset,
    ):
        self.client = client
        self.info = info
        self._metadata = {
            metadatum.key: metadatum.value for metadatum in info.metadata
        }

    @property
    def id(
        self,
    ):
        return self.info.id

    @property
    def name(
        self,
    ):
        return self.info.name

    @property
    def metadata(
        self,
    ) -> Dict[str, Any]:
        return self._metadata

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        **kwargs,
    ):
        # Create the dataset on server side first to get ID info
        ds = schemas.Dataset(
            name=name,
            metadata=[],
        )
        for key in kwargs:
            ds.metadata.append(
                schemas.Metadatum(
                    key=key,
                    value=kwargs[key],
                )
            )
        resp = client._requests_post_rel_host("datasets", json=asdict(ds))

        # @TODO: Handle this response
        if resp:
            pass

        # Retrive newly created dataset with its ID
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        resp = client._requests_get_rel_host(f"datasets/{name}").json()
        metadata = [
            schemas.Metadatum(
                key=metadatum["key"],
                value=metadatum["value"],
            )
            for metadatum in resp["metadata"]
        ]
        info = schemas.Dataset(
            name=resp["name"],
            id=resp["id"],
            metadata=metadata,
        )
        return cls(
            client=client,
            info=info,
        )

    def add_metadatum(self, metadatum: schemas.Metadatum):
        # @TODO: Add endpoint to allow adding custom metadatums
        self.info.metadata.append(metadatum)
        self.__metadata__[metadatum.key] = metadatum

    def add_groundtruth(
        self,
        groundtruth: schemas.GroundTruth,
    ):
        try:
            assert isinstance(groundtruth, schemas.GroundTruth)
        except AssertionError:
            raise TypeError(f"Invalid type `{type(groundtruth)}`")

        if len(groundtruth.annotations) == 0:
            warnings.warn(
                f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations. Skipping..."
            )
            return

        groundtruth.datum.dataset = self.info.name
        self.client._requests_post_rel_host(
            "groundtruths",
            json=asdict(groundtruth),
        )

    def get_groundtruth(self, uid: str) -> schemas.GroundTruth:
        resp = self.client._requests_get_rel_host(
            f"groundtruths/dataset/{self.info.name}/datum/{uid}"
        ).json()
        return schemas.GroundTruth(**resp)

    def get_labels(
        self,
    ) -> List[schemas.Label]:
        labels = self.client._requests_get_rel_host(
            f"labels/dataset/{self.name}"
        ).json()

        return [
            schemas.Label(key=label["key"], value=label["value"])
            for label in labels
        ]

    def get_datums(
        self,
    ) -> List[schemas.Datum]:
        """Returns a list of datums."""
        datums = self.client._requests_get_rel_host(
            f"data/dataset/{self.name}"
        ).json()
        return [schemas.Datum(**datum) for datum in datums]

    def get_images(
        self,
    ) -> List[schemas.ImageMetadata]:
        """Returns a list of Image Metadata if it exists, otherwise raises Dataset contains no images."""
        return [
            schemas.ImageMetadata.from_datum(datum)
            for datum in self.get_datums()
            if schemas.ImageMetadata.valid(datum)
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
    def __init__(
        self,
        client: Client,
        info: schemas.Model,
    ):
        self.client = client
        self.info = info
        self._metadata = {
            metadatum.key: metadatum.value for metadatum in info.metadata
        }

    @property
    def id(
        self,
    ):
        return self.info.id

    @property
    def name(
        self,
    ):
        return self.info.name

    @property
    def metadata(
        self,
    ) -> Dict[str, Any]:
        return self._metadata

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        **kwargs,
    ):
        # Create the dataset on server side first to get ID info
        md = schemas.Model(
            name=name,
            metadata=[],
        )
        for key in kwargs:
            md.metadata.append(
                schemas.Metadatum(
                    key=key,
                    value=kwargs[key],
                )
            )
        resp = client._requests_post_rel_host("models", json=asdict(md))

        # @TODO: Handle this response
        if resp:
            pass

        # Retrive newly created dataset with its ID
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        resp = client._requests_get_rel_host(f"models/{name}").json()
        metadata = [
            schemas.Metadatum(
                key=metadatum["key"],
                value=metadatum["value"],
            )
            for metadatum in resp["metadata"]
        ]
        info = schemas.Model(
            name=resp["name"],
            id=resp["id"],
            metadata=metadata,
        )
        return cls(
            client=client,
            info=info,
        )

    def get_evaluation_status(
        self,
        job_id: int,
    ) -> State:
        resp = self.client._requests_get_rel_host(
            f"evaluations/{job_id}"
        ).json()

        return resp

    def delete(
        self,
    ):
        self.client._requests_delete_rel_host(f"models/{self.name}").json()
        del self

    def add_metadatum(self, metadatum: schemas.Metadatum):
        # @TODO: Add endpoint to allow adding custom metadatums
        self.info.metadata.append(metadatum)
        self.__metadata__[metadatum.key] = metadatum

    def add_prediction(self, prediction: schemas.Prediction):
        try:
            assert isinstance(prediction, schemas.Prediction)
        except AssertionError:
            raise TypeError(
                f"Expected `velour.schemas.Prediction`, got `{type(prediction)}`"
            )

        if len(prediction.annotations) == 0:
            warnings.warn(
                f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations. Skipping..."
            )
            return

        prediction.model = self.info.name
        return self.client._requests_post_rel_host(
            "predictions",
            json=asdict(prediction),
        )

    def get_prediction(self, datum: schemas.Datum) -> schemas.Prediction:
        resp = self.client._requests_get_rel_host(
            f"predictions/model/{self.info.name}/dataset/{datum.dataset}/datum/{datum.uid}",
        ).json()
        return schemas.Prediction(**resp)

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

    def evaluate_detection(
        self,
        dataset: "Dataset",
        annotation_type: AnnotationType = AnnotationType.BOX,
        iou_thresholds_to_compute: List[float] = None,
        iou_thresholds_to_keep: List[float] = None,
        min_area: float = None,
        max_area: float = None,
        label_key: Optional[str] = None,
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

        geometric_area_filters = []
        if min_area:
            geometric_area_filters.append(
                schemas.NumericFilter(
                    value=min_area,
                    operator=">=",
                ),
            )
        if max_area:
            geometric_area_filters.append(
                schemas.NumericFilter(
                    value=max_area,
                    operator="<=",
                ),
            )
        geometric_filters = []
        if geometric_area_filters:
            geometric_filters.append(
                schemas.GeometricAnnotationFilter(
                    annotation_type=annotation_type,
                    area=geometric_area_filters,
                )
            )
        filters = schemas.Filter(
            annotations=schemas.AnnotationFilter(
                annotation_types=[annotation_type],
                geometry=geometric_filters,
            ),
            labels=schemas.LabelFilter(keys=[label_key]),
        )

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
            resp[k] = [schemas.Label(**la) for la in resp[k]]

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

    def get_labels(
        self,
    ) -> List[schemas.Label]:
        labels = self.client._requests_get_rel_host(
            f"labels/model/{self.name}"
        ).json()

        return [
            schemas.Label(key=label["key"], value=label["value"])
            for label in labels
        ]

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
from velour.metatypes import ImageMetadata
from velour.schemas.filters import BinaryExpression, DeclarativeMapper, Filter
from velour.schemas.metadata import validate_metadata


class ClientException(Exception):
    pass


class Client:
    """
    Client for interacting with the velour backend.

    Attributes
    ----------
    host : str
        The host to connect to. Should start with "http://" or "https://".
    access_token : str
        The access token for the host (if the host requires authentication).
    """

    def __init__(self, host: str, access_token: str = None):
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
        """
        Gets the users e-mail address (in the case when auth is enabled)
        or returns None in the case of a no-auth backend.
        """
        resp = self._requests_get_rel_host("user").json()
        return resp["email"]

    def _requests_wrapper(
        self, method_name: str, endpoint: str, *args, **kwargs
    ):
        """
        Wrapper for handling API requests.
        """
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
        """
        Helper for handling POST requests.
        """
        return self._requests_wrapper(
            method_name="post", endpoint=endpoint, *args, **kwargs
        )

    def _requests_get_rel_host(self, endpoint: str, *args, **kwargs):
        """
        Helper for handling GET requests.
        """
        return self._requests_wrapper(
            method_name="get", endpoint=endpoint, *args, **kwargs
        )

    def _requests_put_rel_host(self, endpoint: str, *args, **kwargs):
        """
        Helper for handling PUT requests.
        """
        return self._requests_wrapper(
            method_name="put", endpoint=endpoint, *args, **kwargs
        )

    def _requests_delete_rel_host(self, endpoint: str, *args, **kwargs):
        """
        Helper for handling DELETE requests.
        """
        return self._requests_wrapper(
            method_name="delete", endpoint=endpoint, *args, **kwargs
        )

    def get_bulk_evaluations(
        self,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
    ) -> List[dict]:
        """
        Returns all metrics associated with user-supplied dataset and/or model names.

        Parameters
        ----------
        models : Union[str, List[str], None]
            A list of dataset names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.
        datasets : Union[str, List[str], None]
            A list of model names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.

        Returns
        -------
        List[dict]
            List of dictionaries describing the returned evaluations.

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
        """
        Get all of the datasets associated with `Client`.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the datasets attributed to the `Client` object.
        """
        return self._requests_get_rel_host("datasets").json()

    def get_models(
        self,
    ) -> List[dict]:
        """
        Get all of the models associated with `Client`.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the models attributed to the `Client` object.
        """
        return self._requests_get_rel_host("models").json()

    def get_labels(
        self,
    ) -> List[Label]:
        """
        Get all of the labels associated with `Client`.

        Returns
        ------
        List[Label]
            A list of `Label` objects attributed to `Client`.
        """
        return self._requests_get_rel_host("labels").json()

    def delete_dataset(self, name: str, timeout: int = 0) -> None:
        """
        Delete a dataset using FastAPI's BackgroundProcess

        Parameters
        ----------
        name : str
            The name of the dataset to be deleted.
        timeout : int
            The number of seconds to wait in order to confirm that the dataset was deleted.
        """
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

    def delete_model(self, name: str, timeout: int = 0) -> None:
        """
        Delete a model using FastAPI's BackgroundProcess

        Parameters
        ----------
        name : str
            The name of the model to be deleted.
        timeout : int
            The number of seconds to wait in order to confirm that the model was deleted.
        """
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

    def get_dataset_status(
        self,
        dataset_name: str,
    ) -> State:
        """
        Get the state of a given dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset we want to fetch the state of.

        Returns
        ------
        State
            The state of the `Dataset`.
        """
        try:
            resp = self._requests_get_rel_host(
                f"datasets/{dataset_name}/status"
            ).json()
        except Exception:
            resp = State.NONE

        return resp

    def get_evaluation_status(
        self,
        job_id: int,
    ) -> State:
        """
        Get the state of a given job ID.

        Parameters
        ----------
        job_id : int
            The job id of the evaluation that we want to fetch the state of.

        Returns
        ------
        State
            The state of the `Evaluation`.
        """
        return self._requests_get_rel_host(
            f"evaluations/{job_id}/status"
        ).json()


class Evaluation:
    """
    An object for storing the returned results of a model evaluation (where groundtruths are compared with predictions to measure performance).

    Attributes
    ----------
    client : Client
        The `Client` object associated with the session.
    job_id : int
        The ID of the evaluation job.
    dataset : str
        The name of the dataset.
    model : str
        The name of the model.
    """

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
        """
        The ID of the evaluation job.

        Returns
        ----------
        int:
            The evaluation id.
        """
        return self._id

    @property
    def settings(
        self,
    ) -> schemas.EvaluationJob:
        """
        The settings associated with the evaluation job.

        Returns
        ----------
        schemas.EvaluationJob
            An `EvaluationJob` object describing the evaluation's configuration.
        """
        return self._settings

    @property
    def status(
        self,
    ) -> str:
        """
        The status of the evaluation job.

        Returns
        ----------
        str
            A status (e.g., "done").
        """
        resp = self._client._requests_get_rel_host(
            f"evaluations/{self._id}/status"
        ).json()
        return JobStatus(resp)

    @property
    def task_type(
        self,
    ) -> enums.TaskType:
        """
        The task type of the evaluation job.

        Returns
        ----------
        enums.TaskType
            The task type associated with the `Evaluation` object.
        """
        return self._settings.task_type

    @property
    def results(
        self,
    ) -> schemas.EvaluationResult:
        """
        The results of the evaluation job.

        Returns
        ----------
        schemas.EvaluationResult
            The results from the evaluation.
        """
        result = self._client._requests_get_rel_host(
            f"evaluations/{self._id}"
        ).json()
        return schemas.EvaluationResult(**result)

    def wait_for_completion(
        self, *, interval: float = 1.0, timeout: int = None
    ):
        """
        Runs timeout logic to check when an evaluation is completed.

        Parameters
        ----------
        interval : float
            The number of seconds to waits between retries
        timeout : int
            The total number of seconds to wait for the job to finish.


        Raises
        ----------
        TimeoutError
            If the job's status doesn't change to DONE or FAILED before the timeout expires
        """
        if timeout:
            timeout_counter = int(math.ceil(timeout / interval))
        while self.status not in [JobStatus.DONE, JobStatus.FAILED]:
            time.sleep(interval)
            if timeout:
                timeout_counter -= 1
                if timeout_counter < 0:
                    raise TimeoutError


class Dataset:
    """
    A class describing a given dataset.

    Attributes
    ----------
    client : Client
        The `Client` object associated with the session.
    id : int
        The ID of the dataset.
    name : str
        The name of the dataset.
    metadata : dict
        A dictionary of metadata that describes the dataset.
    geospatial :  dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.
    """

    name = DeclarativeMapper("dataset_names", str)
    metadata = DeclarativeMapper("dataset_metadata", Union[int, float, str])
    geospatial = DeclarativeMapper(
        "dataset_geospatial",
        Union[
            List[List[List[List[Union[float, int]]]]],
            List[List[List[Union[float, int]]]],
            List[Union[float, int]],
            str,
        ],
    )

    def __init__(self):
        self.client: Client = None
        self.id: int = None
        self.name: str = None
        self.metadata: dict = None
        self.geospatial: dict = None

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        geospatial: Dict[
            str,
            Union[
                List[List[List[List[Union[float, int]]]]],
                List[List[List[Union[float, int]]]],
                List[Union[float, int]],
                str,
            ],
        ] = None,
        id: Union[int, None] = None,
    ):
        """
        Create a new `Dataset` object.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the dataset.
        metadata : dict
            A dictionary of metadata that describes the dataset.
        geospatial :  dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.


        Returns
        ----------
        Dataset
           The newly-created `Dataset`.
        """
        dataset = cls()
        dataset.client = client
        dataset.name = name
        dataset.metadata = metadata
        dataset.geospatial = geospatial
        dataset.id = id
        dataset._validate()
        client._requests_post_rel_host("datasets", json=dataset.dict())
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        """
        Fetches a given dataset from the backend.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the dataset.


        Returns
        ----------
        Dataset
            The requested `Dataset`.
        """
        resp = client._requests_get_rel_host(f"datasets/{name}").json()
        dataset = cls()
        dataset.client = client
        dataset.name = resp["name"]
        dataset.metadata = resp["metadata"]
        dataset.geospatial = resp["geospatial"]
        dataset.id = resp["id"]
        dataset._validate()
        return dataset

    def _validate(self):
        """
        Validates the arguments used to create a `Dataset` object.
        """
        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("`id` should be of type `int`")
        if not self.metadata:
            self.metadata = {}
        if not self.geospatial:
            self.geospatial = {}
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        """
        Defines how a `Dataset` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Dataset's` attributes.
        """
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
            "geospatial": self.geospatial,
        }

    def add_groundtruth(
        self,
        groundtruth: GroundTruth,
    ):
        """
        Add a groundtruth to a given dataset.

        Parameters
        ----------
        groundtruth : Groundtruth
            The `Groundtruth` object to add to the `Dataset`.
        """
        if not isinstance(groundtruth, GroundTruth):
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
        """
        Fetches a given groundtruth from the backend.

        Parameters
        ----------
        uid : str
            The UID of the groundtruth to fetch.


        Returns
        ----------
        Groundtruth
            The requested `Groundtruth`.
        """
        resp = self.client._requests_get_rel_host(
            f"groundtruths/dataset/{self.name}/datum/{uid}"
        ).json()
        return GroundTruth(**resp)

    def get_labels(
        self,
    ) -> List[Label]:
        """
        Get all labels associated with a given dataset.

        Returns
        ----------
        List[Label]
            A list of `Labels` associated with the dataset.
        """
        labels = self.client._requests_get_rel_host(
            f"labels/dataset/{self.name}"
        ).json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def get_datums(
        self,
    ) -> List[Datum]:
        """
        Get all datums associated with a given dataset.

        Returns
        ----------
        List[Datum]
            A list of `Datums` associated with the dataset.
        """
        datums = self.client._requests_get_rel_host(
            f"data/dataset/{self.name}"
        ).json()
        return [Datum(**datum) for datum in datums]

    def get_images(
        self,
    ) -> List[ImageMetadata]:
        """
        Get all image metadata associated with a given dataset.

        Returns
        ----------
        List[ImageMetadata]
            A list of `ImageMetadata` associated with the dataset.
        """
        return [
            ImageMetadata.from_datum(datum)
            for datum in self.get_datums()
            if ImageMetadata.valid(datum)
        ]

    def get_evaluations(
        self,
    ) -> List[Evaluation]:
        """
        Get all evaluations associated with a given dataset.

        Returns
        ----------
        List[Evaluation]
            A list of `Evaluations` associated with the dataset.
        """
        model_evaluations = self.client._requests_get_rel_host(
            f"evaluations/dataset/{self.name}"
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
        """
        Finalize the `Dataset` object such that new groundtruths cannot be added to it.
        """
        return self.client._requests_put_rel_host(
            f"datasets/{self.name}/finalize"
        )

    def delete(
        self,
    ):
        """
        Delete the `Dataset` object from the backend.
        """
        self.client._requests_delete_rel_host(f"datasets/{self.name}").json()
        del self


class Model:
    """
    A class describing a model that was trained on a particular dataset.

    Attributes
    ----------
    client : Client
        The `Client` object associated with the session.
    id : int
        The ID of the model.
    name : str
        The name of the model.
    metadata : dict
        A dictionary of metadata that describes the model.
    geospatial :  dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the model.
    """

    name = DeclarativeMapper("models_names", str)
    metadata = DeclarativeMapper("models_metadata", Union[int, float, str])
    geospatial = DeclarativeMapper(
        "model_geospatial",
        Union[
            List[List[List[List[Union[float, int]]]]],
            List[List[List[Union[float, int]]]],
            List[Union[float, int]],
            str,
        ],
    )

    def __init__(self):
        self.client: Client = None
        self.id: int = None
        self.name: str = ""
        self.metadata: dict = None
        self.geospatial: dict = None

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        geospatial: Dict[
            str,
            Union[
                List[List[List[List[Union[float, int]]]]],
                List[List[List[Union[float, int]]]],
                List[Union[float, int]],
                str,
            ],
        ] = None,
        id: Union[int, None] = None,
    ):
        """
        Create a new Model

        Attributes
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the model.
        metadata : dict
            A dictionary of metadata that describes the model.
        geospatial :  dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the model.
        id : int
            The ID of the model.


        Returns
        ----------
        Model
            The newly-created `Model` object.
        """
        model = cls()
        model.client = client
        model.name = name
        model.metadata = metadata
        model.geospatial = geospatial
        model.id = id
        model._validate()
        client._requests_post_rel_host("models", json=model.dict())
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        """
        Fetches a given model from the backend.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the model.


        Returns
        ----------
        Model
            The requested `Model`.
        """
        resp = client._requests_get_rel_host(f"models/{name}").json()
        model = cls()
        model.client = client
        model.name = resp["name"]
        model.metadata = resp["metadata"]
        model.geospatial = resp["geospatial"]
        model.id = resp["id"]
        model._validate()
        return model

    def _validate(self):
        """
        Validates the arguments used to create a `Model` object.
        """
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("`id` should be of type `int`")
        if not self.metadata:
            self.metadata = {}
        if not self.geospatial:
            self.geospatial = {}
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        """
        Defines how a `Model` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Model's` attributes.
        """
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
            "geospatial": self.geospatial,
        }

    def add_prediction(self, prediction: Prediction):
        """
        Add a prediction to a given model.

        Parameters
        ----------
        prediction : Prediction
            The `Prediction` object to add to the `Model`.
        """
        if not isinstance(prediction, Prediction):
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
        """
        Finalize the `Model` object such that new predictions cannot be added to it.
        """
        return self.client._requests_put_rel_host(
            f"models/{self.name}/datasets/{dataset.name}/finalize"
        ).json()

    def evaluate_classification(
        self,
        dataset: Dataset,
        filters: Union[Dict, List[BinaryExpression]] = None,
        timeout: Optional[int] = None,
    ) -> Evaluation:
        """
        Start a classification evaluation job.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate against.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.
        timeout : int
            The number of seconds to wait for the job to finish. Used to ensure deterministic behavior when testing.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """

        # If list[BinaryExpression], convert to filter object
        if not isinstance(filters, dict) and filters is not None:
            filters = Filter.create(filters)

        evaluation = schemas.EvaluationJob(
            model=self.name,
            dataset=dataset.name,
            task_type=enums.TaskType.CLASSIFICATION.value,
            settings=schemas.EvaluationSettings(
                filters=filters,
            ),
        )

        resp = self.client._requests_post_rel_host(
            "evaluations", json=asdict(evaluation)
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
        """
        Start a object-detection evaluation job.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate against.
        iou_threshold_to_compute : List[float]
            Thresholds to compute mAP against.
        iou_thresholds_to_keep : List[float]
            Thresholds to return AP for. Must be subset of `iou_thresholds_to_compute`.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.
        timeout : int
            The number of seconds to wait for the job to finish. Used to ensure deterministic behavior when testing.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """

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

        if not isinstance(filters, dict) and filters is not None:
            filters = Filter.create(filters)

        evaluation = schemas.EvaluationJob(
            model=self.name,
            dataset=dataset.name,
            task_type=enums.TaskType.DETECTION.value,
            settings=schemas.EvaluationSettings(
                parameters=parameters,
                filters=filters,
            ),
        )

        resp = self.client._requests_post_rel_host(
            "evaluations", json=asdict(evaluation)
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
        self,
        dataset: Dataset,
        filters: Union[Dict, List[BinaryExpression]] = None,
        timeout: Optional[int] = None,
    ) -> Evaluation:
        """
        Start a semantic-segmentation evaluation job.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate against.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.
        timeout : int
            The number of seconds to wait for the job to finish. Used to ensure deterministic behavior when testing.

        Returns
        -------
        Evaluation
            a job object that can be used to track the status of the job and get the metrics of it upon completion
        """

        # if list[BinaryExpression], convert to filter object
        if not isinstance(filters, dict) and filters is not None:
            filters = Filter.create(filters)

        # create evaluation job
        evaluation = schemas.EvaluationJob(
            model=self.name,
            dataset=dataset.name,
            task_type=enums.TaskType.SEGMENTATION.value,
            settings=schemas.EvaluationSettings(
                filters=filters,
            ),
        )
        resp = self.client._requests_post_rel_host(
            "evaluations",
            json=asdict(evaluation),
        ).json()

        # create client-side evaluation handler
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
        """
        Delete the `Model` object from the backend.
        """
        self.client._requests_delete_rel_host(f"models/{self.name}").json()
        del self

    def get_prediction(self, datum: Datum) -> Prediction:
        """
        Fetch a particular prediction.

        Parameters
        ----------
        datum : Datum
            The `Datum` of the predictino you want to return

        Returns
        ----------
        Prediction
            The requested `Prediction`.
        """
        resp = self.client._requests_get_rel_host(
            f"predictions/model/{self.name}/dataset/{datum.dataset}/datum/{datum.uid}",
        ).json()
        return Prediction(**resp)

    def get_labels(
        self,
    ) -> List[Label]:
        """
        Get all labels associated with a given model.

        Returns
        ----------
        List[Label]
            A list of `Labels` associated with the model.
        """
        labels = self.client._requests_get_rel_host(
            f"labels/model/{self.name}"
        ).json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def get_evaluations(
        self,
    ) -> List[Evaluation]:
        """
        Get all evaluations associated with a given model.

        Returns
        ----------
        List[Evaluation]
            A list of `Evaluations` associated with the model.
        """
        dataset_evaluations = self.client._requests_get_rel_host(
            f"evaluations/model/{self.name}"
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
    ) -> dict:
        """
        Get all metrics associated with a Model and return them in a `spd.DataFrame`.

        Returns
        ----------
        dict
            A dictionary of the `Model's` metrics and settings.
        """
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
                for metric in evaluation.results.metrics
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

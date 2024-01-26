import os
import logging
import time
import json
import warnings
import requests
from packaging import version
from dataclasses import asdict, dataclass
from typing import Callable, List, Dict, Tuple, Optional, TypeVar, Union
from urllib.parse import urlencode, urljoin

from velour import __version__ as client_version
from velour.enums import TableStatus, AnnotationType, EvaluationStatus, TaskType
from velour.schemas import (
    Label,
    Annotation,
    Datum,
    GroundTruth,
    Prediction,
    EvaluationParameters,
    EvaluationRequest,
    dump_metadata,
    load_metadata,
    validate_metadata,
    DatasetSummary,
    Filter,
)
from velour.schemas.constraints import (
    BinaryExpression,
    StringMapper, 
    GeospatialMapper, 
    DictionaryMapper,
)
from velour.types import MetadataType, GeoJSONType

T = TypeVar("T")


def wait_for_predicate(
    update_func: Callable[[], T],
    pred: Callable[[T], bool],
    timeout: Optional[int],
    interval: float = 1.0,
) -> T:
    """Waits for a condition to become true.

    Repeatedly calls `update_func` to retrieve a new value and checks if the
    condition `pred` is satisfied.  If `pred` is not satisfied within `timeout`
    seconds, raises a TimeoutError.  Polls every `interval` seconds.

    Parameters
    ----------
    update_func:
        A callable that returns a value of type T.
    pred:
        A predicate callable that takes an argument of type T and returns a boolean.
    timeout:
        The maximum number of seconds to wait for the condition to become
        true. If None, waits indefinitely.
    interval:
        The time in seconds between consecutive calls to `update_func`.

    Returns
    ------
    T
        The final value for which `pred` returned True.

    Raises
    ----------
    TimeoutError
        If the condition is not met within `timeout` seconds.

    """
    t_start = time.time()
    state = update_func()
    while not pred(state):
        time.sleep(interval)
        if timeout and time.time() - t_start > timeout:
            raise TimeoutError
        state = update_func()
    return state


def _validate_version(client_version: str, api_version: str):
    """Log and/or warn users if the Velour Python client version differs from the API version."""

    def _msg(state):
        return (
            f"The Velour client version ({client_version}) is {state} than the Velour API version {api_version}"
            f"\t==========================================================================================\n"
            f"\t== Running with a mismatched client != API version may have unexpected results.\n"
            f"\t== Please update your client to \033[1;{api_version}\033[0;31m to avoid aberrant behavior.\n"
            f"\t==========================================================================================\n"
            f"\033[0m"
        )

    if not api_version:
        logging.warning("The Velour API didn't return a version number.")
    elif not client_version:
        logging.warning("The Velour client isn't versioned.")
    elif api_version == client_version:
        logging.debug(
            f"The Velour API version {api_version} matches client version {client_version}."
        )
    elif version.parse(api_version) < version.parse(client_version):
        logging.warning(_msg("newer"))
    else:
        logging.warning(_msg("older"))


@dataclass
class _DatumSchema(Datum):
    dataset_name: Optional[str] = None
    
    def strip(self) -> Datum:
        attr = self.__dict__.copy()
        attr.pop("dataset_name")
        return Datum(**attr)


@dataclass
class _GroundtruthSchema(GroundTruth):
    datum: _DatumSchema
    annotations: List[Annotation]

    def strip(self) -> GroundTruth:
        return GroundTruth(
            datum=self.datum.strip(),
            annotations=self.annotations,
        )


@dataclass
class _PredictionSchema(Prediction):
    model_name: Optional[str] = None
    datum: _DatumSchema
    annotations: List[Annotation]
    
    def strip(self) -> Prediction:
        attr = self.__dict__.copy()
        attr.pop("model_name")
        return Prediction(**attr)


class ClientException(Exception):
    def __init__(self, resp):
        self.status_code = resp.status_code
        self.detail = resp.json()["detail"]
        super().__init__(str(self.detail))


class Client:
    """
    Velour client object for interacting with the api.

    Parameters
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

        # check the connection by getting the api version number
        api_version = self._get_api_version_number()

        _validate_version(
            client_version=client_version, api_version=api_version
        )

        success_str = f"Successfully connected to host at {self.host}"
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

    def _get_api_version_number(
        self,
    ) -> Union[str, None]:
        """
        Gets the version number of the API.
        """
        resp = self._requests_get_rel_host("api-version").json()
        return resp["api_version"]

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
                raise ClientException(resp)
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

    def get_labels(
        self,
        filters: Filter = None,
    ) -> List[dict]:
        """
        Get labels associated with `Client`.

        Parameters
        ----------
        filters : Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[Label]
            A list of `Label` objects attributed to `Client`.
        """
        kwargs = {}
        if filters:
            kwargs["json"] = asdict(filters)
        return self._requests_get_rel_host("labels", **kwargs).json()

    def create_dataset(
        self,
        dataset: dict,
    ):
        """
        Creates a dataset.

        Parameters
        ----------
        dataset : dict
            A dictionary describing dataset attributes. See `velour.coretypes.Dataset` for reference.
        """
        self._requests_post_rel_host("datasets", json=dataset)

    def get_dataset(
        self,
        name: str,
    ) -> dict:
        """
        Gets a dataset by name.

        Parameters
        ----------
        name : str
            The name of the dataset to fetch.

        Returns
        -------
        dict
            A dictionary containing all of the associated dataset attributes.
        """
        try:
            return self._requests_get_rel_host(f"datasets/{name}").json()
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_datasets(
        self,
        filters: Filter = None,
    ) -> List[dict]:
        """
        Get datasets associated with `Client`.

        Parameters
        ----------
        filters : Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the datasets attributed to the `Client` object.
        """
        kwargs = {}
        if filters:
            kwargs["json"] = asdict(filters)
        return self._requests_get_rel_host("datasets", **kwargs).json()

    def get_datums(
        self,
        filters: Filter = None,
    ) -> List[dict]:
        """
        Get datums associated with `Client`.

        Parameters
        ----------
        filters : Filter, optional
            Optional filter to constrain by.

        Returns
        -------
        List[dict]
            A list of dictionaries describing all the datums of the specified dataset.
        """
        kwargs = {}
        if filters:
            kwargs["json"] = asdict(filters)
        return self._requests_get_rel_host("data", **kwargs).json()

    def get_dataset_status(
        self,
        dataset_name: str,
    ) -> TableStatus:
        """
        Get the state of a given dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset we want to fetch the state of.

        Returns
        ------
        TableStatus | None
            The state of the `Dataset`. Returns None if dataset does not exist.
        """
        try:
            resp = self._requests_get_rel_host(
                f"datasets/{dataset_name}/status"
            ).json()
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e
        return TableStatus(resp)

    def get_dataset_summary(self, dataset_name: str) -> dict:
        return self._requests_get_rel_host(
            f"datasets/{dataset_name}/summary"
        ).json()

    def delete_dataset(self, name: str, timeout: int = 0) -> None:
        """
        Delete a dataset using FastAPI's `BackgroundProcess`.

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
                if self.get_dataset(name) is None:
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "Dataset wasn't deleted within timeout interval"
                )

    def create_model(
        self,
        model: dict,
    ):
        """
        Creates a model.

        Parameters
        ----------
        model : dict
            A dictionary describing model attributes. See `velour.coretypes.Model` for reference.
        """
        self._requests_post_rel_host("models", json=model)

    def get_model(
        self,
        name: str,
    ) -> dict:
        """
        Gets a model by name.

        Parameters
        ----------
        name : str
            The name of the model to fetch.

        Returns
        -------
        dict
            A dictionary containing all of the associated model attributes.
        """
        try:
            return self._requests_get_rel_host(f"models/{name}").json()
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_models(
        self,
        filters: Filter = None,
    ) -> List[dict]:
        """
        Get models associated with `Client`.

        Parameters
        ----------
        filters : Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the models attributed to the `Client` object.
        """
        kwargs = {}
        if filters:
            kwargs["json"] = asdict(filters)
        return self._requests_get_rel_host("models", **kwargs).json()

    def get_model_status(
        self,
        dataset_name: str,
        model_name: str,
    ) -> TableStatus:
        """
        Get the state of a given model.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset that the model is operating over.
        model_name : str
            The name of the model we want to fetch the state of.

        Returns
        ------
        TableStatus
            The state of the `Model`.
        """
        try:
            resp = self._requests_get_rel_host(
                f"models/{model_name}/dataset/{dataset_name}/status"
            ).json()
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e
        return TableStatus(resp)

    def delete_model(self, name: str, timeout: int = 0) -> None:
        """
        Delete a model using FastAPI's `BackgroundProcess`.

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
                if self.get_model(name) is None:
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "Model wasn't deleted within timeout interval"
                )

    def get_evaluations(
        self,
        *,
        evaluation_ids: Union[int, List[int], None] = None,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
    ) -> List[dict]:
        """
        Returns all metrics associated with user-supplied dataset and/or model names.

        Parameters
        ----------
        evaluation_ids : Union[int, List[int], None]
            A list of job ids to return metrics for.  If the user passes a single value, it will automatically be converted to a list for convenience.
        models : Union[str, List[str], None]
            A list of model names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.
        datasets : Union[str, List[str], None]
            A list of dataset names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.

        Returns
        -------
        List[dict]
            List of dictionaries describing the returned evaluations.

        """

        if not (evaluation_ids or models or datasets):
            raise ValueError(
                "Please provide at least one evaluation_id, model name, or dataset name"
            )

        def _build_query_param(param_name, element, typ):
            """Parse `element` to a list of `typ`, return a dict that can be urlencoded."""
            if not element:
                return {}
            if isinstance(element, typ):
                element = [element]
            return {param_name: ",".join(map(str, element))}

        params = {
            **_build_query_param("evaluation_ids", evaluation_ids, int),
            **_build_query_param("models", models, str),
            **_build_query_param("datasets", datasets, str),
        }

        query_str = urlencode(params)
        endpoint = f"evaluations?{query_str}"

        return self._requests_get_rel_host(endpoint).json()

    def evaluate(self, req: EvaluationRequest) -> List[dict]:
        """
        Creates as many evaluations as necessary to fulfill the request.

        Parameters
        ----------
        req : schemas.EvaluationRequest
            The requested evaluation parameters.

        Returns
        -------
        List[schemas.EvaluationResponse]
            A list of evaluations that meet the parameters.
        """
        resp = self._requests_post_rel_host(
            "evaluations", json=asdict(req)
        ).json()
        return resp


class Evaluation:
    """
    Wraps `velour.client.Job` to provide evaluation-specifc members.
    """

    def __init__(self, client: Client, *_, **kwargs):
        """
        Defines important attributes of the API's `EvaluationResult`.

        Attributes
        ----------
        id : int
            The id of the evaluation.
        model_name : str
            The name of the evaluated model.
        datum_filter : schemas.Filter
            The filter used to select the datums for evaluation.
        status : EvaluationStatus
            The status of the evaluation.
        metrics : List[dict]
            A list of metric dictionaries returned by the job.
        confusion_matrices : List[dict]
            A list of confusion matrix dictionaries returned by the job.
        """
        self.client = client
        self._from_dict(**kwargs)

    def dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "datum_filter": asdict(self.datum_filter),
            "parameters": asdict(self.parameters),
            "status": self.status.value,
            "metrics": self.metrics,
            "confusion_matrices": self.confusion_matrices,
            **self.kwargs,
        }

    def _from_dict(
        self,
        *_,
        id: int,
        model_name: str,
        datum_filter: Filter,
        parameters: EvaluationParameters,
        status: EvaluationStatus,
        metrics: List[dict],
        confusion_matrices: List[dict],
        **kwargs,
    ):
        self.id = id
        self.model_name = model_name
        self.datum_filter = (
            Filter(**datum_filter)
            if isinstance(datum_filter, dict)
            else datum_filter
        )
        self.parameters = (
            EvaluationParameters(**parameters)
            if isinstance(parameters, dict)
            else parameters
        )
        self.status = EvaluationStatus(status)
        self.metrics = metrics
        self.confusion_matrices = confusion_matrices
        self.kwargs = kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

    def poll(self) -> EvaluationStatus:
        """
        Poll the backend.

        Updates the evaluation with the latest state from the backend.

        Returns
        -------
        enums.EvaluationStatus
            The status of the evaluation.

        Raises
        ----------
        ClientException
            If an Evaluation with the given `evaluation_id` is not found.
        """
        response = self.client.get_evaluations(evaluation_ids=[self.id])
        if not response:
            raise ClientException("Not Found")
        self._from_dict(**response[0])
        return self.status

    def wait_for_completion(
        self,
        *,
        timeout: int = None,
        interval: float = 1.0,
    ) -> EvaluationStatus:
        """
        Blocking function that waits for evaluation to finish.

        Parameters
        ----------
        timeout : int, optional
            Length of timeout in seconds.
        interval : float, default=1.0
            Polling interval in seconds.
        """
        t_start = time.time()
        while self.poll() not in [
            EvaluationStatus.DONE,
            EvaluationStatus.FAILED,
        ]:
            time.sleep(interval)
            if timeout and time.time() - t_start > timeout:
                raise TimeoutError
        return self.status

    def to_dataframe(
        self,
        stratify_by: Tuple[str, str] = None,
    ):
        """
        Get all metrics associated with a Model and return them in a `pd.DataFrame`.

        Returns
        ----------
        pd.DataFrame
            Evaluation metrics being displayed in a `pd.DataFrame`.

        Raises
        ------
        ModuleNotFoundError
            This function requires the use of `pandas.DataFrame`.

        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Must have pandas installed to use `get_metric_dataframes`."
            )

        if not stratify_by:
            column_type = "evaluation"
            column_name = self.id
        else:
            column_type = stratify_by[0]
            column_name = stratify_by[1]

        metrics = [
            {**metric, column_type: column_name} for metric in self.metrics
        ]
        df = pd.DataFrame(metrics)
        for k in ["label", "parameters"]:
            df[k] = df[k].fillna("n/a")
        df["parameters"] = df["parameters"].apply(json.dumps)
        df["label"] = df["label"].apply(
            lambda x: f"{x['key']}: {x['value']}" if x != "n/a" else x
        )
        df = df.pivot(
            index=["type", "parameters", "label"], columns=[column_type]
        )
        return df


class Dataset:
    """
    A class describing a given dataset.

    Attribute
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

    name = StringMapper("dataset_names")
    metadata = DictionaryMapper("dataset_metadata")
    geospatial = GeospatialMapper("dataset_geospatial")

    def __init__(
        self,
        client: Client,
        name: str,
        metadata: Optional[MetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
        id: Union[int, None] = None,
        delete_if_exists: bool = False,
        **_,
    ):
        """
        Create or get a `Dataset` object.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the dataset.
        metadata : dict
            A dictionary of metadata that describes the dataset.
        geospatial : dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.
        delete_if_exists : bool, default=False
            Deletes any existing dataset with the same name.
        """
        self.name = name
        self.metadata = metadata
        self.geospatial = geospatial
        self.id = id
        self._validate()

        if delete_if_exists and client.get_dataset(name) is not None:
            client.delete_dataset(name, timeout=30)

        if delete_if_exists or client.get_dataset(name) is None:
            client.create_dataset(self.dict())

        for k, v in client.get_dataset(name).items():
            setattr(self, k, v)
        self.client = client

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

        # metadata
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

    def __str__(self):
        return str(self.dict())

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
            "metadata": dump_metadata(self.metadata),
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
        groundtruth : GroundTruth
            The `GroundTruth` object to add to the `Dataset`.
        """
        if not isinstance(groundtruth, GroundTruth):
            raise TypeError(f"Invalid type `{type(groundtruth)}`")

        if len(groundtruth.annotations) == 0:
            warnings.warn(
                f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations."
            )

        groundtruth = _GroundtruthSchema(**groundtruth)
        groundtruth.datum.dataset_name = self.name
        self.client._requests_post_rel_host(
            "groundtruths",
            json=asdict(groundtruth),
        )

    def get_groundtruth(self, datum: Union[Datum, str]) -> GroundTruth:
        """
        Fetches a given groundtruth from the backend.

        Parameters
        ----------
        datum : Datum
            The Datum of the 'GroundTruth' to fetch.


        Returns
        ----------
        GroundTruth
            The requested `GroundTruth`.
        """
        uid = datum.uid if isinstance(datum, Datum) else datum
        resp = self.client._requests_get_rel_host(
            f"groundtruths/dataset/{self.name}/datum/{uid}"
        ).json()
        return _GroundtruthSchema(**resp).strip()

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

    def get_datums(self) -> List[Datum]:
        """
        Get all datums associated with a given dataset.

        Returns
        ----------
        List[Datum]
            A list of `Datums` associated with the dataset.
        """
        datums = self.client.get_datums(
            filters=Filter(dataset_names=[self.name])
        )
        return [Datum(**datum) for datum in datums]

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
        return [
            Evaluation(self.client, **resp)
            for resp in self.client.get_evaluations(datasets=self.name)
        ]

    def get_summary(self) -> DatasetSummary:
        """
        Get the summary of a given dataset.

        Returns
        -------
        DatasetSummary
            The summary of the dataset. This class has the following fields:

            name: name of the dataset

            num_datums: total number of datums in the dataset

            num_annotations: total number of labeled annotations in the dataset. if an
            object (such as a bounding box) has multiple labels then each label is counted separately

            num_bounding_boxes: total number of bounding boxes in the dataset

            num_polygons: total number of polygons in the dataset

            num_groundtruth_multipolygons: total number of multipolygons in the dataset

            num_rasters: total number of rasters in the dataset

            task_types: list of the unique task types in the dataset

            labels: list of the unique labels in the dataset

            datum_metadata: list of the unique metadata dictionaries in the dataset that are associated
            to datums

            groundtruth_annotation_metadata: list of the unique metadata dictionaries in the dataset that are
            associated to annotations
        """
        resp = self.client.get_dataset_summary(self.name)
        return DatasetSummary(**resp)

    def finalize(
        self,
    ):
        """
        Finalize the `Dataset` object such that new `GroundTruths` cannot be added to it.
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


class Model:
    """
    A class describing a model that was trained on a particular dataset.

    Attribute
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

    name = StringMapper("model_names")
    metadata = DictionaryMapper("model_metadata")
    geospatial = GeospatialMapper("model_geospatial")

    def __init__(
        self,
        client: Client,
        name: str,
        metadata: Optional[MetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
        id: Union[int, None] = None,
        delete_if_exists: bool = False,
    ):
        """
        Create or get a `Model` object.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the model.
        metadata : dict
            A dictionary of metadata that describes the model.
        geospatial : dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the model.
        id : int, optional
            SQL index for model.
        delete_if_exists : bool, default=False
            Deletes any existing model with the same name.
        """
        self.name = name
        self.metadata = metadata
        self.geospatial = geospatial
        self.id = id
        self._validate()

        if delete_if_exists and client.get_model(name) is not None:
            client.delete_model(name, timeout=30)

        if delete_if_exists or client.get_model(name) is None:
            client.create_model(self.dict())

        for k, v in client.get_model(name).items():
            setattr(self, k, v)
        self.client = client

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

        # metadata
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

    def __str__(self):
        return str(self.dict())

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
            "metadata": dump_metadata(self.metadata),
            "geospatial": self.geospatial,
        }

    def add_prediction(
        self, dataset: Union[Dataset, str], prediction: Prediction
    ):
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
                f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations."
            )

        prediction = _PredictionSchema(**prediction, model_name=self.name)
        prediction.datum.dataset_name = dataset.name if isinstance(dataset, Dataset) else dataset
        return self.client._requests_post_rel_host(
            "predictions",
            json=asdict(prediction),
        )
    
    def get_prediction(self, dataset: Dataset, datum: Datum) -> Prediction:
        """
        Fetch a particular prediction.

        Parameters
        ----------
        datum : Union[Datum, str]
            The `Datum` or datum UID of the prediction to return.

        Returns
        ----------
        Prediction
            The requested `Prediction`.
        """
        resp = self.client._requests_get_rel_host(
            f"predictions/model/{self.name}/dataset/{dataset.name}/datum/{datum.uid}",
        ).json()
        return _PredictionSchema(**resp).strip()

    def finalize_inferences(self, dataset: "Dataset") -> None:
        """
        Finalize the `Model` object such that new `Predictions` cannot be added to it.
        """
        return self.client._requests_put_rel_host(
            f"models/{self.name}/datasets/{dataset.name}/finalize"
        ).json()

    def _format_filters(
        self,
        datasets: Union[Dataset, List[Dataset]],
        filters: Union[Dict, List[BinaryExpression]],
    ) -> Union[dict, Filter]:
        """Formats evaluation request's `datum_filter` input."""

        # get list of dataset names
        dataset_names_from_obj = []
        if isinstance(datasets, list):
            dataset_names_from_obj = [dataset.name for dataset in datasets]
        elif isinstance(datasets, Dataset):
            dataset_names_from_obj = [datasets.name]

        # format filtering object
        if isinstance(filters, list) or filters is None:
            filters = filters if filters else []
            filters = Filter.create(filters)

            # reset model name
            filters.model_names = None
            filters.model_geospatial = None
            filters.model_metadata = None

            # set dataset names
            if not filters.dataset_names:
                filters.dataset_names = []
            filters.dataset_names.extend(dataset_names_from_obj)

        elif isinstance(filters, dict):
            # reset model name
            filters["model_names"] = None
            filters["model_geospatial"] = None
            filters["model_metadata"] = None

            # set dataset names
            if (
                "dataset_names" not in filters
                or filters["dataset_names"] is None
            ):
                filters["dataset_names"] = []
            filters["dataset_names"].extend(dataset_names_from_obj)

        return filters

    def evaluate_classification(
        self,
        datasets: Union[Dataset, List[Dataset]] = None,
        filters: Union[Dict, List[BinaryExpression]] = None,
    ) -> Evaluation:
        """
        Start a classification evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """
        if not datasets and not filters:
            raise ValueError(
                "Evaluation requires the definition of either datasets, dataset filters or both."
            )

        datum_filter = self._format_filters(datasets, filters)

        evaluation = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=EvaluationParameters(task_type=TaskType.CLASSIFICATION),
        )
        resp = self.client.evaluate(evaluation)
        if len(resp) != 1:
            raise RuntimeError
        resp = resp[0]

        # resp should have keys "missing_pred_keys", "ignored_pred_keys", with values
        # list of label dicts. convert label dicts to Label objects

        evaluation_job = Evaluation(
            client=self.client,
            **resp,
        )

        return evaluation_job

    def evaluate_detection(
        self,
        datasets: Union[Dataset, List[Dataset]] = None,
        filters: Union[Dict, List[BinaryExpression]] = None,
        convert_annotations_to_type: AnnotationType = None,
        iou_thresholds_to_compute: List[float] = None,
        iou_thresholds_to_return: List[float] = None,
    ) -> Evaluation:
        """
        Start a object-detection evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filters : Union[Dict, List[BinaryExpression]], optional
            Optional set of filters to constrain evaluation by.
        convert_annotations_to_type : enums.AnnotationType, optional
            Forces the object detection evaluation to compute over this type.
        iou_thresholds_to_compute : List[float], optional
            Thresholds to compute mAP against.
        iou_thresholds_to_return : List[float], optional
            Thresholds to return AP for. Must be subset of `iou_thresholds_to_compute`.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """
        if iou_thresholds_to_compute is None:
            iou_thresholds_to_compute = [
                round(0.5 + 0.05 * i, 2) for i in range(10)
            ]
        if iou_thresholds_to_return is None:
            iou_thresholds_to_return = [0.5, 0.75]

        parameters = EvaluationParameters(
            task_type=TaskType.DETECTION,
            convert_annotations_to_type=convert_annotations_to_type,
            iou_thresholds_to_compute=iou_thresholds_to_compute,
            iou_thresholds_to_return=iou_thresholds_to_return,
        )

        datum_filter = self._format_filters(datasets, filters)

        evaluation = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=parameters,
        )
        resp = self.client.evaluate(evaluation)
        if len(resp) != 1:
            raise RuntimeError
        resp = resp[0]

        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects

        evaluation_job = Evaluation(
            client=self.client,
            **resp,
        )

        return evaluation_job

    def evaluate_segmentation(
        self,
        datasets: Union[Dataset, List[Dataset]] = None,
        filters: Union[Dict, List[BinaryExpression]] = None,
    ) -> Evaluation:
        """
        Start a semantic-segmentation evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.

        Returns
        -------
        Evaluation
            a job object that can be used to track the status of the job and get the metrics of it upon completion
        """

        datum_filter = self._format_filters(datasets, filters)

        # create evaluation job
        evaluation = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=EvaluationParameters(task_type=TaskType.SEGMENTATION),
        )
        resp = self.client.evaluate(evaluation)
        if len(resp) != 1:
            raise RuntimeError
        resp = resp[0]

        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects

        # create client-side evaluation handler
        evaluation_job = Evaluation(
            client=self.client,
            **resp,
        )

        return evaluation_job

    def delete(
        self,
    ):
        """
        Delete the `Model` object from the backend.
        """
        self.client._requests_delete_rel_host(f"models/{self.name}").json()

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
        return [
            Evaluation(self.client, **resp)
            for resp in self.client.get_evaluations(models=self.name)
        ]

    def get_metric_dataframes(
        self,
    ) -> dict:
        """
        Get all metrics associated with a Model and return them in a `pd.DataFrame`.

        Returns
        ----------
        dict
            A dictionary of the `Model's` metrics and settings, with the metrics being displayed in a `pd.DataFrame`.
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
                for metric in evaluation.metrics
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

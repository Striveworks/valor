import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Union
from urllib.parse import urlencode, urljoin

import requests
from packaging import version

from velour import __version__ as client_version
from velour.enums import TableStatus
from velour.exceptions import (
    ClientAlreadyConnectedError,
    ClientException,
    ClientNotConnectedError,
)
from velour.schemas import EvaluationRequest
from velour.types import T

_connection = None


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


@dataclass
class ClientConnection:
    """
    Velour client object for interacting with the api.

    Parameters
    ----------
    host : str
        The host to connect to. Should start with "http://" or "https://".
    access_token : str
        The access token for the host (if the host requires authentication).
    """

    host: str
    access_token: Optional[str] = None

    def __post_init__(self):
        if not (
            self.host.startswith("http://") or self.host.startswith("https://")
        ):
            raise ValueError(
                f"host must stat with 'http://' or 'https://' but got {self.host}"
            )

        if not self.host.endswith("/"):
            self.host += "/"
        self.access_token = os.getenv("VELOUR_ACCESS_TOKEN", self.access_token)

        # check the connection by getting the api version number
        api_version = self.get_api_version()
        """version = se."""

        self._validate_version(
            client_version=client_version, api_version=api_version
        )

        success_str = f"Successfully connected to host at {self.host}"
        print(success_str)

    def _validate_version(self, client_version: str, api_version: str):
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

    def create_groundtruths(
        self,
        groundtruth: dict,
    ) -> None:
        """
        Create a groundtruth.

        `CREATE` endpoint.

        Parameters
        ----------
        groundtruth : dict
            The groundtruth to be created.
        """
        return self._requests_post_rel_host(
            "groundtruths",
            json=groundtruth,
        )

    def get_groundtruth(
        self,
        dataset_name: str,
        datum_uid: str,
    ) -> dict:
        """
        Get a particular groundtruth.

        `GET` endpoint.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset the datum belongs to.
        datum_uid : str
            The uid of the desired datum.

        Returns
        ----------
        dict
            The requested groundtruth.
        """
        return self._requests_get_rel_host(
            f"groundtruths/dataset/{dataset_name}/datum/{datum_uid}"
        ).json()

    def create_predictions(
        self,
        prediction: dict,
    ) -> None:
        """
        Create a prediction.

        `CREATE` endpoint.

        Parameters
        ----------
        prediction : dict
            The prediction to be created.
        """
        return self._requests_post_rel_host(
            "predictions",
            json=prediction,
        )

    def get_prediction(
        self,
        dataset_name: str,
        model_name: str,
        datum_uid: str,
    ) -> dict:
        """
        Get a particular prediction.

        `GET` endpoint.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset the datum belongs to.
        model_name : str
            The name of the model that made the prediction.
        datum_uid : str
            The uid of the desired datum.

        Returns
        ----------
        dict
            The requested prediction.
        """
        return self._requests_get_rel_host(
            f"predictions/model/{model_name}/dataset/{dataset_name}/datum/{datum_uid}",
        ).json()

    def get_labels(
        self,
        filter_: Optional[dict] = None,
    ) -> List[dict]:
        """
        Gets all labels with option to filter.

        `GET` endpoint.
        """
        kwargs = {}
        if filter_:
            kwargs["json"] = filter_
        return self._requests_get_rel_host("labels", **kwargs).json()

    def get_labels_from_dataset(self, name: str) -> List[dict]:
        """
        Get all labels associated with a dataset's groundtruths.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to search by.

        Returns
        ------
        List[dict]
            A list of labels.
        """
        return self._requests_get_rel_host(f"labels/dataset/{name}").json()

    def get_labels_from_model(self, name: str) -> List[dict]:
        """
        Get all labels associated with a model's predictions.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the model to search by.

        Returns
        ------
        List[dict]
            A list of labels.
        """
        return self._requests_get_rel_host(f"labels/model/{name}").json()

    def create_dataset(self, dataset: dict):
        """
        Creates a dataset.

        `CREATE` endpoint.

        Parameters
        ----------
        dataset : dict
            A dictionary describing dataset attributes. See `velour.coretypes.Dataset` for reference.
        """
        self._requests_post_rel_host("datasets", json=dataset)

    def get_datasets(self, filter_: Optional[dict] = None) -> List[dict]:
        """
        Get all datasets with option to filter.

        `GET` endpoint.

        Parameters
        ----------
        filter_ : Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the datasets attributed to the `Client` object.
        """
        kwargs = {}
        if filter_:
            kwargs["json"] = filter_
        return self._requests_get_rel_host("datasets", **kwargs).json()

    def get_dataset(self, name: str) -> dict:
        """
        Gets a dataset by name.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to fetch.

        Returns
        -------
        dict
            A dictionary containing all of the associated dataset attributes.
        """
        return self._requests_get_rel_host(f"datasets/{name}").json()

    def get_dataset_status(self, name: str) -> TableStatus:
        """
        Get the state of a given dataset.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset we want to fetch the state of.

        Returns
        ------
        TableStatus
            The state of the dataset.
        """
        resp = self._requests_get_rel_host(f"datasets/{name}/status").json()
        return TableStatus(resp)

    def get_dataset_summary(self, name: str) -> dict:
        """
        Gets the summary of a dataset.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to create a summary for.

        Returns
        -------
        dict
            A dictionary containing the dataset summary.
        """
        return self._requests_get_rel_host(f"datasets/{name}/summary").json()

    def finalize_dataset(self, name: str) -> None:
        """
        Finalizes a dataset such that new groundtruths cannot be added to it.

        `PUT` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset.
        """
        return self._requests_put_rel_host(f"datasets/{name}/finalize")

    def delete_dataset(self, name: str) -> None:
        """
        Deletes a dataset.

        `DELETE` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to be deleted.
        """
        self._requests_delete_rel_host(f"datasets/{name}")

    def get_datums(self, filter_: Optional[dict] = None) -> List[dict]:
        """
        Get datums associated with `Client`.

        `GET` endpoint.

        Parameters
        ----------
        filter_ : dict, optional
            Optional filter to constrain by.

        Returns
        -------
        List[dict]
            A list of dictionaries describing all the datums of the specified dataset.
        """
        kwargs = {}
        if filter_:
            kwargs["json"] = filter_
        return self._requests_get_rel_host("data", **kwargs).json()

    def get_datum(self):
        """
        `GET` endpoint.
        """
        pass

    def create_model(self, model: dict) -> None:
        """
        Creates a model.

        `CREATE` endpoint.

        Parameters
        ----------
        model : dict
            A dictionary describing model attributes. See `velour.coretypes.Model` for reference.
        """
        self._requests_post_rel_host("models", json=model)

    def get_models(self, filter_: Optional[dict] = None) -> List[dict]:
        """
        Get all models with option to filter results.

        `GET` endpoint.

        Parameters
        ----------
        filter_ : Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the models.
        """
        kwargs = {}
        if filter_:
            kwargs["json"] = filter_
        return self._requests_get_rel_host("models", **kwargs).json()

    def get_model(self, name: str) -> dict:
        """
        Gets a model by name.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the model to fetch.

        Returns
        -------
        dict
            A dictionary containing all of the associated model attributes.
        """
        return self._requests_get_rel_host(f"models/{name}").json()

    def get_model_eval_requests(self):
        """
        `GET` endpoint.
        """
        pass

    def get_model_status(
        self,
        dataset_name: str,
        model_name: str,
    ) -> TableStatus:
        """
        Get the state of a given model over a dataset.

        `GET` endpoint.

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
        resp = self._requests_get_rel_host(
            f"models/{model_name}/dataset/{dataset_name}/status"
        ).json()
        return TableStatus(resp)

    def finalize_inferences(
        self,
        dataset_name: str,
        model_name: str,
    ) -> None:
        """
        Finalizes a model-dataset pairing such that new prediction cannot be added to it.

        `PUT` endpoint.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        model_name : str
            The name of the model.
        """
        return self._requests_put_rel_host(
            f"models/{model_name}/datasets/{dataset_name}/finalize"
        ).json()

    def delete_model(self, name: str) -> None:
        """
        Deletes a model.

        `DELETE` endpoint.

        Parameters
        ----------
        name : str
            The name of the model to be deleted.
        """
        self._requests_delete_rel_host(f"models/{name}")

    def evaluate(self, request: EvaluationRequest) -> List[dict]:
        """
        Creates as many evaluations as necessary to fulfill the request.

        `CREATE` endpoint.

        Parameters
        ----------
        request : schemas.EvaluationRequest
            The requested evaluation parameters.

        Returns
        -------
        List[dict]
            A list of evaluations that meet the parameters.
        """
        return self._requests_post_rel_host(
            "evaluations", json=asdict(request)
        ).json()

    def get_evaluations(
        self,
        *,
        evaluation_ids: Optional[List[int]] = None,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Returns all evaluations associated with user-supplied dataset and/or model names.

        `GET` endpoint.

        Parameters
        ----------
        evaluation_ids : List[int], optional
            A list of job ids to return metrics for.
        models : List[str], optional
            A list of model names that we want to return metrics for.
        datasets : List[str], optional
            A list of dataset names that we want to return metrics for.

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

    def get_user(self) -> Union[str, None]:
        """
        Gets the users e-mail address (in the case when auth is enabled)
        or returns None in the case of a no-auth backend.

        `GET` endpoint.

        Returns
        -------
        Union[str, None]
            The user's email address or `None` if it doesn't exist.
        """
        resp = self._requests_get_rel_host("user").json()
        return resp["email"]

    def get_api_version(self) -> str:
        """
        Gets the version number of the API.

        `GET` endpoint.

        Returns
        -------
        Union[str, None]
            The api version or `None` if it doesn't exist.
        """
        resp = self._requests_get_rel_host("api-version").json()
        return resp["api_version"]

    def health(self):
        """
        Checks if service is healthy.

        `GET` endpoint.
        """
        pass

    def ready(self):
        """
        Checks if service is ready.

        `GET` endpoint.
        """
        pass


def connect(
    host: str,
    access_token: Optional[str] = None,
    reconnect: bool = False,
):
    """
    Establishes a connection to the client.

    Parameters
    ----------
    host : str
        The host to connect to. Should start with "http://" or "https://".
    access_token : str
        The access token for the host (if the host requires authentication).
    """
    global _connection
    if _connection is not None and not reconnect:
        raise ClientAlreadyConnectedError

    try:
        _connection = ClientConnection(host, access_token)
        _connection.get_api_version()
    except Exception as e:
        _connection = None
        # TODO - raise ClientException showing no service found
        raise e


def get_connection() -> ClientConnection:
    """
    Gets the active client connection.

    Returns
    -------
    ClientConnection
        The active client connection.

    Raises
    ------
    ClientNotConnectedError
        If there is no active connection.
    """
    global _connection
    if _connection is None:
        raise ClientNotConnectedError
    return _connection

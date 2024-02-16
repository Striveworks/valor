import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, TypeVar, Union
from urllib.parse import urlencode, urljoin

import requests
from packaging import version

from velour import __version__ as client_version
from velour.enums import TableStatus
from velour.exceptions import (
    ClientAlreadyConnectedError,
    ClientConnectionFailed,
    ClientException,
    ClientNotConnectedError,
)
from velour.schemas import EvaluationRequest

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


def get_json_size(json_obj: object, encoding: str = "utf-8") -> int:
    """
    Returns the number of bytes to describe the json as a string.

    Parameters
    ----------
    json_obj : object
        A JSON-compatible object.
    encoding : str, default='utf-8'
        The method used to encode the string object into a bytes format.

    Returns
    -------
    int
        The size of the object in bytes.
    """
    return len(
        json.dumps(json_obj, ensure_ascii=False).encode(
            encoding
        )  # this outputs bytes
    )


def chunk_list(
    json_list: list, chunk_size_bytes: int, encoding: str = "utf-8"
) -> List[list]:
    """
    Chunks a list into smaller lists.

    Parameters
    ----------
    json_list : list
        A list of JSON-compatible objects.
    chunk_size_bytes : int
        The maximum number of bytes that a multi-element array can use.
    encoding : str, default='utf-8'
        The method used to encode the string object into a bytes format.

    Returns
    -------
    List[list]
        A list of lists containing JSON-compatible objects.
    """
    # edge case
    if len(json_list) == 1:
        if get_json_size(json_list, encoding) > chunk_size_bytes:
            logging.warning(
                f"Attempting to POST an object that is larger than {chunk_size_bytes} bytes."
            )
        return [json_list]

    n_elements = len(json_list)
    avg_element_size = get_json_size(json_list) / n_elements
    n_elements_per_chunk = int(chunk_size_bytes / avg_element_size)
    n_chunks = math.ceil(n_elements / n_elements_per_chunk)

    chunks = []
    for i in range(n_chunks):
        lhi = i * n_elements_per_chunk
        rhi = (i + 1) * n_elements_per_chunk
        new_chunk_size = get_json_size(
            json_obj=json_list[lhi:rhi], encoding=encoding
        )
        if (
            new_chunk_size > chunk_size_bytes
        ):  # Recursively chunk if still too large.
            chunks.extend(
                chunk_list(
                    json_list=json_list[lhi:rhi],
                    chunk_size_bytes=chunk_size_bytes,
                    encoding=encoding,
                )
            )
        else:
            chunks.append(json_list[lhi:rhi])
    return chunks


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

    Raises
    ------
        ClientConnectionFailed:
            If a connection could not be established.
    """

    host: str
    access_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

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
        self.username = self.username or os.getenv("VELOUR_USERNAME")
        self.password = self.password or os.getenv("VELOUR_PASSWORD")

        if self.username and self.password and self.access_token:
            raise ValueError(
                "You can only provide either a username and password or an access token, not both."
            )

        if self.username and self.password:
            self._using_username_password = True
            self._get_access_token_from_username_and_password()
        else:
            self._using_username_password = False

        # check the connection by getting the api version number
        try:
            api_version = self.get_api_version()
        except Exception as e:
            raise ClientConnectionFailed(str(e))

        self._validate_version(
            client_version=client_version, api_version=api_version
        )

        success_str = f"Successfully connected to host at {self.host}"
        print(success_str)

    def _get_access_token_from_username_and_password(self) -> None:
        """Sets the access token from the username and password."""
        resp = self._requests_post_rel_host(
            "token",
            ignore_auth=True,
            data={"username": self.username, "password": self.password},
        )
        if resp.ok:
            self.access_token = resp.json()

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
        self,
        method_name: str,
        endpoint: str,
        ignore_auth: bool = False,
        *args,
        **kwargs,
    ):
        """
        Wrapper for handling API requests.

        Parameters
        ----------
        method_name : str
            The name of the method to use for the request.
        endpoint : str
            The endpoint to send the request to.
        ignore_auth : bool, default=False
            Option to ignore authentication when you know the endpoint does not
            require a bearer token. this is used by the `_get_access_token_from_username_and_password`
            to avoid infinite recursion.
        """
        accepted_methods = ["get", "post", "put", "delete"]
        if method_name not in accepted_methods:
            raise ValueError(
                f"method_name should be one of {accepted_methods}"
            )

        if endpoint[0] == "/":
            raise ValueError(
                "`endpoint` should not start with a forward slash."
            )

        url = urljoin(self.host, endpoint)
        requests_method = getattr(requests, method_name)

        tried = False
        while True:
            if self.access_token is not None:
                headers = {"Authorization": f"Bearer {self.access_token}"}
            else:
                headers = None
            resp = requests_method(url, headers=headers, *args, **kwargs)
            if not resp.ok:
                # check if unauthorized and if using username and password, get a new
                # token and try the request again
                if (
                    resp.status_code in [401, 403]
                    and self._using_username_password
                    and not tried
                    and not ignore_auth
                ):
                    self._get_access_token_from_username_and_password()
                else:
                    try:
                        raise ClientException(resp)
                    except (requests.exceptions.JSONDecodeError, KeyError):
                        resp.raise_for_status()
            else:
                break
            tried = True

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
        groundtruths: List[dict],
        chunk_size_bytes: int = int(1e7),
    ) -> None:
        """
        Creates groundtruths.

        `CREATE` endpoint.

        Parameters
        ----------
        groundtruths : List[dict]
            The groundtruths to be created.
        chunk_size_bytes : int, default=1e7
            Maximum size of a POST'ed json in bytes. Defaults to 10MB.
        """
        chunked_groundtruths = chunk_list(
            json_list=groundtruths,
            chunk_size_bytes=chunk_size_bytes,
        )
        for chunk in chunked_groundtruths:
            self._requests_post_rel_host(
                "groundtruths",
                json=chunk,
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
        predictions: List[dict],
        chunk_size_bytes: int = int(1e7),
    ) -> None:
        """
        Creates predictions.

        `CREATE` endpoint.

        Parameters
        ----------
        predictions : List[dict]
            The predictions to be created.
        chunk_size_bytes : int, default=1e7
            Maximum size of a POST'ed json in bytes. Defaults to 10MB.
        """
        chunked_predictions = chunk_list(
            json_list=predictions,
            chunk_size_bytes=chunk_size_bytes,
        )
        for chunk in chunked_predictions:
            self._requests_post_rel_host(
                "predictions",
                json=chunk,
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
        Get all datums using an optional filter

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

    def get_datum(
        self,
        dataset_name: str,
        uid: str,
    ) -> dict:
        """
        Get datum.
        `GET` endpoint.
        Parameters
        ----------
        dataset_name : str
            The dataset the datum belongs to.
        uid : str
            The UID of the datum.
        Returns
        -------
        dict
            A dictionary describing a datum.
        """
        return self._requests_get_rel_host(
            f"/data/dataset/{dataset_name}/uid/{uid}"
        ).json()

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

    def get_model_eval_requests(self, name: str) -> List[dict]:
        """
        Get all evaluations that have been created for a model.

        This does not return evaluation results.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the model.

        Returns
        -------
        List[dict]
            A list of evaluations.
        """
        return self._requests_get_rel_host(
            f"/models/{name}/eval-requests"
        ).json()

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
        Finalizes a model-dataset pairing such that new predictions cannot be added to it.

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

    def health(self) -> str:
        """
        Checks if service is healthy.

        `GET` endpoint.
        """
        resp = self._requests_get_rel_host("user").json()
        return resp["status"]

    def ready(self) -> str:
        """
        Checks if service is ready.

        `GET` endpoint.
        """
        resp = self._requests_get_rel_host("user").json()
        return resp["status"]


def _create_connection():
    """
    Creates and manages a connection to the Velour API.

    This function initializes a connection closure that can be used to establish and retrieve a client connection to the Velour API. It returns two functions: `connect` and `get_connection`.

    The `connect` function is used to establish a new connection to the API, either with a new host or by reconnecting to an existing host. It raises an error if a connection is already established and `reconnect` is not set to `True`.

    The `get_connection` function is used to retrieve the current active connection. It raises an error if there's no active connection.

    Returns
    -------
    tuple
        (connect, get_connection)
    """
    _connection = None

    def connect(
        host: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        access_token: Optional[str] = None,
        reconnect: bool = False,
    ):
        """
        Establishes a connection to the Velour API.

        Parameters
        ----------
        host : str
            The host to connect to. Should start with "http://" or "https://".
        username: str
            The username for the host (if the host requires authentication).
        password: str
            The password for the host (if the host requires authentication).
        access_token : str
            The access token for the host (if the host requires authentication). Mutually
            exclusive with `username` and `password`.

        Raises
        ------
        ClientAlreadyConnectedError:
            If the connection has previously been established.
        ClientConnectionFailed:
            If a connection could not be established.
        """

        nonlocal _connection
        if _connection is not None and not reconnect:
            raise ClientAlreadyConnectedError
        _connection = ClientConnection(
            host,
            username=username,
            password=password,
            access_token=access_token,
        )

    def get_connection():
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
        if _connection is None:
            raise ClientNotConnectedError
        return _connection

    def reset_connection():
        """
        Resets the connection to its initial state.
        """
        nonlocal _connection
        _connection = None

    return connect, get_connection, reset_connection


connect, get_connection, reset_connection = _create_connection()

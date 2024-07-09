import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TypeVar, Union
from urllib.parse import urlencode, urljoin

import requests
from packaging import version

from valor import __version__ as client_version
from valor.enums import TableStatus
from valor.exceptions import (
    ClientAlreadyConnectedError,
    ClientConnectionFailed,
    ClientNotConnectedError,
    raise_client_exception,
)

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


def _format_request_timeout(
    timeout: Optional[float], default: float
) -> Optional[float]:
    """
    Converts user-requested timeout into requests library format to avoid issues with passing `None` to `requests.timeout`.

    Parameters
    ----------
    timeout : float, optional
        The user requested timeout in seconds. Timeout <= 0 disables the timeout.
    default : float
        The default timeout in seconds. Used when user timeout is None.

    Returns
    -------
    float | None
        The validated request timeout.
    """
    if timeout is None:
        return default
    elif timeout <= 0:
        return None
    else:
        return timeout


@dataclass
class ClientConnection:
    """
    Valor client object for interacting with the api.

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
        self.access_token = os.getenv("VALOR_ACCESS_TOKEN", self.access_token)
        self.username = self.username or os.getenv("VALOR_USERNAME")
        self.password = self.password or os.getenv("VALOR_PASSWORD")

        if self.username and self.password and self.access_token:
            raise ValueError(
                "You can only provide either a username and password or an access token, not both."
            )

        if self.username and self.password:
            self._using_username_password = True
            self._get_access_token_from_username_and_password()
        else:
            self._using_username_password = False

        if self.access_token or self._using_username_password:
            self._using_auth = True
        else:
            self._using_auth = False

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
        """Log and/or warn users if the Valor Python client version differs from the API version."""

        def _msg(state):
            return (
                f"The Valor client version ({client_version}) is {state} than the Valor API version {api_version}"
                f"\t==========================================================================================\n"
                f"\t== Running with a mismatched client != API version may have unexpected results.\n"
                f"\t== Please update your client to \033[1;{api_version}\033[0;31m to avoid aberrant behavior.\n"
                f"\t==========================================================================================\n"
                f"\033[0m"
            )

        if not api_version:
            logging.warning("The Valor API didn't return a version number.")
        elif not client_version:
            logging.warning("The Valor client isn't versioned.")
        elif api_version == client_version:
            logging.debug(
                f"The Valor API version {api_version} matches client version {client_version}."
            )
        elif version.parse(api_version) < version.parse(client_version):
            logging.warning(_msg("newer"))
        else:
            logging.warning(_msg("older"))

    def _requests_wrapper(
        self,
        *_,
        method_name: str,
        endpoint: str,
        timeout: Optional[float],
        ignore_auth: bool = False,
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
        timeout : float, optional
            An optional request timeout in seconds.
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

        if not ignore_auth and self._using_auth:
            # get a new token
            if self._using_username_password:
                self._get_access_token_from_username_and_password()
            headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            headers = None

        try:
            resp = requests_method(
                url,
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
        except requests.exceptions.ReadTimeout as e:
            if timeout:
                raise TimeoutError(
                    f"Client request timed out at {timeout} seconds."
                )
            else:
                raise e

        if not resp.ok:
            raise_client_exception(resp)

        return resp

    def _requests_post_rel_host(
        self,
        endpoint: str,
        *_,
        json: Union[dict, list, None] = None,
        params: Union[dict, list, None] = None,
        data: Optional[dict] = None,
        ignore_auth: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Helper for handling POST requests.

        Parameters
        ----------
        endpoint : str
            The endpoint to send the request to.
        json : dict | list, optional
            An optional kwarg to pass to the request.
        params : dict, optional
            An optional kwarg to pass to the request.
        data : dict, optional
            An optional kwarg to pass to the request.
        ignore_auth : bool, default=False
            Option to ignore authentication when you know the endpoint does not
            require a bearer token. This is used by the `_get_access_token_from_username_and_password`
            to avoid infinite recursion.
        timeout : float, optional
            An optional request timeout in seconds.
        """
        timeout = _format_request_timeout(timeout=timeout, default=3000)
        return self._requests_wrapper(
            method_name="post",
            endpoint=endpoint,
            json=json,
            params=params,
            data=data,
            ignore_auth=ignore_auth,
            timeout=timeout,
        )

    def _requests_get_rel_host(
        self,
        endpoint: str,
        *_,
        timeout: Optional[float] = None,
    ):
        """
        Helper for handling GET requests.

        Parameters
        ----------
        endpoint : str
            The endpoint to send the request to.
        timeout : float, optional
            An optional request timeout in seconds.
        """
        timeout = _format_request_timeout(timeout=timeout, default=3000)
        return self._requests_wrapper(
            method_name="get", endpoint=endpoint, timeout=timeout
        )

    def _requests_put_rel_host(
        self,
        endpoint: str,
        *_,
        timeout: Optional[float] = None,
    ):
        """
        Helper for handling PUT requests.

        Parameters
        ----------
        endpoint : str
            The endpoint to send the request to.
        timeout : float, optional
            An optional request timeout in seconds.
        """
        timeout = _format_request_timeout(timeout=timeout, default=3000)
        return self._requests_wrapper(
            method_name="put", endpoint=endpoint, timeout=timeout
        )

    def _requests_delete_rel_host(
        self,
        endpoint: str,
        *_,
        timeout: Optional[float] = None,
    ):
        """
        Helper for handling DELETE requests.

        Parameters
        ----------
        endpoint : str
            The endpoint to send the request to.
        timeout : float, optional
            An optional request timeout in seconds.
        """
        timeout = _format_request_timeout(timeout=timeout, default=3000)
        return self._requests_wrapper(
            method_name="delete", endpoint=endpoint, timeout=timeout
        )

    def create_groundtruths(
        self,
        groundtruths: List[dict],
        *_,
        ignore_existing_datums: bool = False,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Creates ground truths.

        `CREATE` endpoint.

        Parameters
        ----------
        groundtruths : List[dict]
            The ground truths to be created.
        ignore_existing_datums : bool, default=False
            If True, will ignore datums that already exist in the backend.
            If False, will raise an error if any datums already exist.
            Default is False.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        self._requests_post_rel_host(
            endpoint="groundtruths",
            json=groundtruths,
            params={"ignore_existing_datums": ignore_existing_datums},
            timeout=timeout,
        )

    def get_groundtruth(
        self,
        dataset_name: str,
        datum_uid: str,
        *_,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Get a particular ground truth.

        `GET` endpoint.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset the datum belongs to.
        datum_uid : str
            The uid of the desired datum.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ----------
        dict
            The requested ground truth.
        """
        return self._requests_get_rel_host(
            f"groundtruths/dataset/{dataset_name}/datum/{datum_uid}",
            timeout=timeout,
        ).json()

    def create_predictions(
        self,
        predictions: List[dict],
        *_,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Creates predictions.

        `CREATE` endpoint.

        Parameters
        ----------
        predictions : List[dict]
            The predictions to be created.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        self._requests_post_rel_host(
            endpoint="predictions",
            json=predictions,
            timeout=timeout,
        )

    def get_prediction(
        self,
        dataset_name: str,
        model_name: str,
        datum_uid: str,
        *_,
        timeout: Optional[float] = None,
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
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ----------
        dict
            The requested prediction.
        """
        return self._requests_get_rel_host(
            f"predictions/model/{model_name}/dataset/{dataset_name}/datum/{datum_uid}",
            timeout=timeout,
        ).json()

    def get_labels(
        self,
        filters: Optional[dict] = None,
        *_,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Gets all labels using an optional filter.

        `POST` endpoint.

        Parameters
        ----------
        filters : dict, optional
            An optional filter.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        list[dict]
            A list of labels in JSON format.
        """
        filters = filters if filters else dict()
        return self._requests_post_rel_host(
            "labels/filter",
            json=filters,
            timeout=timeout,
        ).json()

    def get_labels_from_dataset(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Get all labels associated with a dataset's ground truths.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to search by.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ------
        List[dict]
            A list of labels.
        """
        return self._requests_get_rel_host(
            f"labels/dataset/{name}",
            timeout=timeout,
        ).json()

    def get_labels_from_model(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Get all labels associated with a model's predictions.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the model to search by.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ------
        List[dict]
            A list of labels.
        """
        return self._requests_get_rel_host(
            f"labels/model/{name}", timeout=timeout
        ).json()

    def create_dataset(
        self,
        dataset: dict,
        *_,
        timeout: Optional[float] = None,
    ):
        """
        Creates a dataset.

        `CREATE` endpoint.

        Parameters
        ----------
        dataset : dict
            A dictionary describing dataset attributes. See `valor.coretypes.Dataset` for reference.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        self._requests_post_rel_host("datasets", json=dataset, timeout=timeout)

    def get_datasets(
        self,
        filters: Optional[dict] = None,
        *_,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Get all datasets with option to filter.

        `POST` endpoint.

        Parameters
        ----------
        filters : Filter, optional
            An optional filter.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ------
        List[dict]
            A list of datasets in JSON format.
        """
        filters = filters if filters else dict()
        return self._requests_post_rel_host(
            "datasets/filter",
            json=filters,
            timeout=timeout,
        ).json()

    def get_dataset(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Gets a dataset by name.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to fetch.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        dict
            A dictionary containing all of the associated dataset attributes.
        """
        return self._requests_get_rel_host(
            f"datasets/{name}", timeout=timeout
        ).json()

    def get_dataset_status(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> TableStatus:
        """
        Get the state of a given dataset.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset we want to fetch the state of.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ------
        TableStatus
            The state of the dataset.
        """
        resp = self._requests_get_rel_host(
            f"datasets/{name}/status", timeout=timeout
        ).json()
        return TableStatus(resp)

    def get_dataset_summary(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Gets the summary of a dataset.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to create a summary for.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        dict
            A dictionary containing the dataset summary.
        """
        return self._requests_get_rel_host(
            f"datasets/{name}/summary", timeout=timeout
        ).json()

    def finalize_dataset(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Finalizes a dataset such that new ground truths cannot be added to it.

        `PUT` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        return self._requests_put_rel_host(
            f"datasets/{name}/finalize", timeout=timeout
        )

    def delete_dataset(
        self, name: str, *_, timeout: Optional[float] = None
    ) -> None:
        """
        Deletes a dataset.

        `DELETE` endpoint.

        Parameters
        ----------
        name : str
            The name of the dataset to be deleted.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        self._requests_delete_rel_host(f"datasets/{name}", timeout=timeout)

    def get_datums(
        self,
        filters: Optional[dict] = None,
        *_,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Get all datums using an optional filter.

        `POST` endpoint.

        Parameters
        ----------
        filters : dict, optional
            An optional filter.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        List[dict]
            A list of datums in JSON format.
        """
        filters = filters if isinstance(filters, dict) else dict()
        return self._requests_post_rel_host(
            "data/filter", json=filters, timeout=timeout
        ).json()

    def get_datum(
        self,
        dataset_name: str,
        uid: str,
        *_,
        timeout: Optional[float] = None,
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
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        dict
            A dictionary describing a datum.
        """
        return self._requests_get_rel_host(
            f"data/dataset/{dataset_name}/uid/{uid}",
            timeout=timeout,
        ).json()

    def create_model(
        self,
        model: dict,
        *_,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Creates a model.

        `CREATE` endpoint.

        Parameters
        ----------
        model : dict
            A dictionary describing model attributes. See `valor.coretypes.Model` for reference.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        self._requests_post_rel_host("models", json=model, timeout=timeout)

    def get_models(
        self,
        filters: Optional[dict] = None,
        *_,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Get all models using an optional filter.

        `POST` endpoint.

        Parameters
        ----------
        filters : Filter, optional
            An optional filter.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ------
        List[dict]
            A list of models in JSON format.
        """
        filters = filters if filters else dict()
        return self._requests_post_rel_host(
            "models/filter",
            json=filters,
            timeout=timeout,
        ).json()

    def get_model(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Gets a model by name.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the model to fetch.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        dict
            A dictionary containing all of the associated model attributes.
        """
        return self._requests_get_rel_host(
            f"models/{name}", timeout=timeout
        ).json()

    def get_model_eval_requests(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Get all evaluations that have been created for a model.

        This does not return evaluation results.

        `GET` endpoint.

        Parameters
        ----------
        name : str
            The name of the model.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        List[dict]
            A list of evaluations.
        """
        return self._requests_get_rel_host(
            f"/models/{name}/eval-requests",
            timeout=timeout,
        ).json()

    def get_model_status(
        self,
        dataset_name: str,
        model_name: str,
        *_,
        timeout: Optional[float] = None,
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
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        ------
        TableStatus
            The state of the `Model`.
        """
        resp = self._requests_get_rel_host(
            f"models/{model_name}/dataset/{dataset_name}/status",
            timeout=timeout,
        ).json()
        return TableStatus(resp)

    def finalize_inferences(
        self,
        dataset_name: str,
        model_name: str,
        *_,
        timeout: Optional[float] = None,
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
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        return self._requests_put_rel_host(
            f"models/{model_name}/datasets/{dataset_name}/finalize",
            timeout=timeout,
        ).json()

    def delete_model(
        self,
        name: str,
        *_,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Deletes a model.

        `DELETE` endpoint.

        Parameters
        ----------
        name : str
            The name of the model to be deleted.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        self._requests_delete_rel_host(f"models/{name}", timeout=timeout)

    def evaluate(
        self,
        request: dict,
        *_,
        allow_retries: bool = False,
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Creates as many evaluations as necessary to fulfill the request.

        `CREATE` endpoint.

        Parameters
        ----------
        request : dict
            The requested evaluation parameters.
        allow_retries : bool, default = False
            Option to retry previously failed evaluations.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        List[dict]
            A list of evaluations that meet the parameters.
        """
        query_str = urlencode({"allow_retries": allow_retries})
        endpoint = f"evaluations?{query_str}"
        return self._requests_post_rel_host(
            endpoint, json=request, timeout=timeout
        ).json()

    def get_evaluations(
        self,
        *,
        evaluation_ids: Optional[List[int]],
        models: Optional[List[str]],
        datasets: Optional[List[str]],
        metrics_to_sort_by: Optional[Dict[str, Union[Dict[str, str], str]]],
        timeout: Optional[float] = None,
    ) -> List[dict]:
        """
        Returns all evaluations associated with user-supplied dataset and/or model names.

        `GET` endpoint.

        Parameters
        ----------
        evaluation_ids : List[int], optional
            A list of job IDs to return metrics for.
        models : List[str], optional
            A list of model names that we want to return metrics for.
        datasets : List[str], optional
            A list of dataset names that we want to return metrics for.
        metrics_to_sort_by: dict[str, str | dict[str, str]], optional
            An optional dict of metric types to sort the evaluations by.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

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
            **_build_query_param(
                "metrics_to_sort_by", json.dumps(metrics_to_sort_by), str
            ),
        }

        query_str = urlencode(params)
        endpoint = f"evaluations?{query_str}"

        return self._requests_get_rel_host(endpoint, timeout=timeout).json()

    def delete_evaluation(
        self,
        evaluation_id: int,
        *_,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Deletes an evaluation.

        `DELETE` endpoint.

        Parameters
        ----------
        evaluation_id : int
            The id of the evaluation to be deleted.
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        self._requests_delete_rel_host(
            f"evaluations/{evaluation_id}", timeout=timeout
        )

    def get_user(
        self,
        *_,
        timeout: Optional[float] = None,
    ) -> Union[str, None]:
        """
        Gets the users e-mail address (in the case when auth is enabled)
        or returns None in the case of a no-auth back end.

        `GET` endpoint.

        Parameters
        ----------
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        Union[str, None]
            The user's email address or `None` if it doesn't exist.
        """
        resp = self._requests_get_rel_host("user", timeout=timeout).json()
        return resp["email"]

    def get_api_version(
        self,
        *_,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Gets the version number of the API.

        `GET` endpoint.

        Parameters
        ----------
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.

        Returns
        -------
        Union[str, None]
            The api version or `None` if it doesn't exist.
        """
        resp = self._requests_get_rel_host(
            "api-version", timeout=timeout
        ).json()
        return resp["api_version"]

    def health(
        self,
        *_,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Checks if service is healthy.

        `GET` endpoint.

        Parameters
        ----------
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        resp = self._requests_get_rel_host("user", timeout=timeout).json()
        return resp["status"]

    def ready(self, *_, timeout: Optional[float] = None) -> str:
        """
        Checks if service is ready.

        `GET` endpoint.

        Parameters
        ----------
        timeout : float, optional
            The number of seconds the client should wait until raising a timeout.
        """
        resp = self._requests_get_rel_host("user", timeout=timeout).json()
        return resp["status"]


def _create_connection():
    """
    Creates and manages a connection to the Valor API.

    This function initializes a connection closure that can be used to establish and retrieve a client connection to the Valor API. It returns two functions: `connect` and `get_connection`.

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
        Establishes a connection to the Valor API.

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

import logging
import os
import time
import json
from typing import Callable, List, Optional, TypeVar, Union
from urllib.parse import urlencode, urljoin

import requests
from packaging import version

from velour import __version__ as client_version
from velour.enums import TableStatus
from velour.schemas.evaluation import EvaluationResult


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


class ClientException(Exception):
    def __init__(self, resp):
        self.status_code = resp.status_code
        self.detail = resp.json()["detail"]
        super().__init__(str(self.detail))


class Client:
    """
    Client for interacting with the velour backend.

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
    ) -> List[dict]:
        """
        Get all of the labels associated with `Client`.

        Returns
        ------
        List[Label]
            A list of `Label` objects attributed to `Client`.
        """
        return self._requests_get_rel_host("labels").json()

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
    ) -> List[dict]:
        """
        Get all of the datasets associated with `Client`.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the datasets attributed to the `Client` object.
        """
        return self._requests_get_rel_host("datasets").json()

    def get_datums(
        self,
        dataset_name: str,
    ) -> List[dict]:
        """
        Get all datums associated with a dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to search over.

        Returns
        -------
        List[dict]
            A list of dictionaries describing all the datums of the specified dataset.
        """
        return self._requests_get_rel_host(
            f"data/dataset/{dataset_name}"
        ).json()

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
                if self.get_dataset_status(name) is None:
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
    ) -> List[dict]:
        """
        Get all of the models associated with `Client`.

        Returns
        ------
        List[dict]
            A list of dictionaries describing all the models attributed to the `Client` object.
        """
        return self._requests_get_rel_host("models").json()

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
                if self.get_dataset_status(name) is None:
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "Model wasn't deleted within timeout interval"
                )

    def get_bulk_evaluations(
        self,
        *,
        evaluation_ids: Union[int, List[int], None] = None,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
    ) -> List[EvaluationResult]:
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

        evals = self._requests_get_rel_host(endpoint).json()
        return [EvaluationResult(**eval) for eval in evals]


class DeletionJob:
    def __init__(
        self,
        client: Client,
        *,
        dataset_name: str = None,
        model_name: str = None,
        **kwargs,
    ):
        self.client = client
        self.dataset_name = dataset_name
        self.model_name = model_name

        if dataset_name and not model_name:
            self.func = self.client.get_dataset
            self.args = self.dataset_name
        elif model_name and not dataset_name:
            self.func = self.client.get_model
            self.args = self.model_name
        else:
            raise ValueError

        for k, v in kwargs.items():
            setattr(self, k, v)

    def completed(self) -> bool:
        return self.func(self.args) is None

    def wait_for_completion(
        self,
        *,
        timeout: int = None,
        interval: float = 1.0,
    ):
        """
        Runs timeout logic to check when an job is completed.

        Parameters
        ----------
        timeout : int
            The total number of seconds to wait for the job to finish.
        interval : float
            The polling interval.

        Raises
        ----------
        TimeoutError
            If the job's status doesn't change to True the timeout expires
        """
        wait_for_predicate(
            lambda: self.completed(),
            lambda status: status is True,
            timeout,
            interval,
        )

import math
import os
import time
from typing import List, Union
from urllib.parse import urljoin

import log
import requests

from velour import __version__ as client_version
from velour import schemas
from velour.enums import JobStatus
from velour.schemas.evaluation import EvaluationResult


def _validate_version(api_version: str):
    """Log and/or warn users if the Velour Python client version differs from the API version."""

    def _msg(state):
        return (
            f"The Velour client version ({client_version}) is {state} than the Velour API version {api_version}"
            f"\t==========================================================================================\n"
            f"\t== Running with a mismatched client != API version may have unexpected results.\n"
            f"\t== Please install \033[1;velour-client=={api_version}\033[0;31m to avoid aberrant behavior.\n"
            f"\t==========================================================================================\n"
            f"\033[0m"
        )

    if not api_version:
        log.warning("Velour returned no version")
    elif api_version == client_version:
        log.debug(
            f"Velour API version {api_version} matches client version {client_version}."
        )
    elif api_version < client_version:
        log.error(_msg("newer"))
    else:
        log.error(_msg("older"))


class ClientException(Exception):
    pass


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

        # check the connection by hitting the users endpoint
        api_version = self._get_api_version_number()

        _validate_version(api_version=api_version)

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

    def get_dataset_status(
        self,
        dataset_name: str,
    ) -> JobStatus:
        """
        Get the state of a given dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset we want to fetch the state of.

        Returns
        ------
        JobStatus
            The state of the `Dataset`.
        """
        resp = self._requests_get_rel_host(
            f"datasets/{dataset_name}/status"
        ).json()
        return JobStatus(resp)

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
                if self.get_dataset_status(name) == JobStatus.NONE:
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "Dataset wasn't deleted within timeout interval"
                )

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
        model_name: str,
    ) -> JobStatus:
        """
        Get the state of a given model.

        Parameters
        ----------
        model_name : str
            The name of the model we want to fetch the state of.

        Returns
        ------
        JobStatus
            The state of the `Model`.
        """
        resp = self._requests_get_rel_host(
            f"models/{model_name}/status"
        ).json()
        return JobStatus(resp)

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
                if self.get_dataset_status(name) == JobStatus.NONE:
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "Model wasn't deleted within timeout interval"
                )

    def get_evaluation_status(
        self,
        evaluation_id: int,
    ) -> JobStatus:
        """
        Get the state of a given job ID.

        Parameters
        ----------
        evaluation_id : int
            The job id of the evaluation that we want to fetch the state of.

        Returns
        ------
        JobStatus
            The state of the `Evaluation`.
        """
        resp = self._requests_get_rel_host(
            f"evaluations/{evaluation_id}/status"
        ).json()
        return JobStatus(resp)

    def get_evaluation(
        self,
        evaluation_id: int,
    ) -> EvaluationResult:
        """
        The results of an evaluation job.

        Returns
        ----------
        schemas.EvaluationResult
            The results from the evaluation.
        """
        result = self._requests_get_rel_host(
            f"evaluations/{evaluation_id}"
        ).json()
        return schemas.EvaluationResult(**result)

    def get_bulk_evaluations(
        self,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
    ) -> List[EvaluationResult]:
        """
        Returns all metrics associated with user-supplied dataset and/or model names.

        Parameters
        ----------
        models : Union[str, List[str], None]
            A list of model names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.
        datasets : Union[str, List[str], None]
            A list of dataset names that we want to return metrics for. If the user passes a string, it will automatically be converted to a list for convenience.

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
        print(evals)
        return [EvaluationResult(**eval) for eval in evals]


class Job:
    def __init__(
        self,
        client: Client,
        *,
        dataset_name: str = None,
        model_name: str = None,
        evaluation_id: int = None,
        **kwargs,
    ):
        self.client = client
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.evaluation_id = evaluation_id

        if evaluation_id:
            self.url = f"evaluations/{evaluation_id}/status"
        elif dataset_name and not model_name:
            self.url = f"datasets/{dataset_name}/status"
        elif model_name and not dataset_name:
            self.url = f"models/{model_name}/status"
        elif model_name and dataset_name:
            raise NotImplementedError(
                "The status endpoint of dataset-model pairings has not been implemented yet."
            )
        else:
            raise ValueError

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def status(self) -> JobStatus:
        resp = self.client._requests_get_rel_host(self.url).json()
        return JobStatus(resp)

    def results(self):
        """
        Certain types of jobs have a return type.
        """
        if self.status == JobStatus.DONE:
            if self.evaluation_id:
                return self.client.get_evaluation(self.evaluation_id)
            else:
                return None

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

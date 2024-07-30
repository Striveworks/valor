import json
from datetime import datetime

from fastapi import HTTPException

from valor_api import enums, logger


class ServiceUnavailable(Exception):
    """
    Raises an exception if the Valor service is unavailble.
    """

    def __init__(self, message: str):
        super().__init__(message)


""" Dataset """


class DatasetAlreadyExistsError(Exception):
    """
    Raises an exception if the user tries to create a dataset with a name that already exists.

    Parameters
    -------
    name : str
        The name of the dataset.
    """

    def __init__(self, name: str):
        super().__init__(f"Dataset with name `{name}` already exists.")


class DatasetDoesNotExistError(Exception):
    """
    Raises an exception if the user tries to manipulate a dataset that doesn't exist.

    Parameters
    -------
    name : str
        The name of the dataset.
    """

    def __init__(self, name: str):
        super().__init__(f"Dataset with name `{name}` does not exist.")


class DatasetEmptyError(Exception):
    """
    Raises an exception if the user tries to finalize an empty dataset.

    Parameters
    -------
    name : str
        The name of the dataset.
    """

    def __init__(self, name: str):
        super().__init__(
            f"cannot finalize dataset `{name}` as it does not contain any data."
        )


class DatasetFinalizedError(Exception):
    """
    Raises an exception if the user tries to add groundtruths to a dataset that has already been finalized.

    Parameters
    -------
    name : str
        The name of the dataset.
    """

    def __init__(self, name: str):
        super().__init__(
            f"cannot edit dataset `{name}` since it has been finalized."
        )


class DatasetNotFinalizedError(Exception):
    """
    Raises an exception if the user tries to process a dataset that hasn't been finalized.

    Parameters
    -------
    name : str
        The name of the dataset.
    """

    def __init__(self, name: str, action: str = "evaluate"):
        super().__init__(
            f"cannot {action} on dataset `{name}` since it has not been finalized."
        )


class DatasetStateError(Exception):
    """
    Raise an exception if a requested state transition is illegal.

    This is a catch-all exception for dataset transitions. If it is called regularly a more specific
    exception should be defined.

    Parameters
    -------
    dataset_name : str
        The name of the dataset.
    current_state : enums.TableStatus
        The current state of the dataset.
    requested_state : enums.TableStatus
        The illegal state transition that was requested for the dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        current_state: enums.TableStatus,
        requested_state: enums.TableStatus,
    ):
        super().__init__(
            f"Dataset `{dataset_name}` attempted an illegal transition from `{current_state}` to `{requested_state}`."
        )


""" Model """


class ModelAlreadyExistsError(Exception):
    """
    Raises an exception if the user tries to create a model using a name that already exists in the database.

    Parameters
    -------
    name : str
        The name of the model.
    """

    def __init__(self, name: str):
        super().__init__(f"Model with name `{name}` already exists.")


class ModelDoesNotExistError(Exception):
    """
    Raises an exception if the user tries to manipulate a model that doesn't exist.

    Parameters
    -------
    name : str
        The name of the model.
    """

    def __init__(self, name: str):
        super().__init__(f"Model with name `{name}` does not exist.")


class ModelFinalizedError(Exception):
    """
    Raises an exception if the user tries to add predictions to a model that has been finalized.

    Parameters
    -------
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    """

    def __init__(self, *, dataset_name: str, model_name: str):
        super().__init__(
            f"cannot edit inferences for model `{model_name}` on dataset `{dataset_name}` since it has been finalized"
        )


class ModelNotFinalizedError(Exception):
    """
    Raises an exception if the user tries to manipulate a model that hasn't been finalized.

    Parameters
    -------
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    """

    def __init__(self, *, dataset_name: str, model_name: str):
        super().__init__(
            f"cannot evaluate inferences for model `{model_name}` on dataset `{dataset_name}` since it has not been finalized."
        )


class ModelStateError(Exception):
    """
    Raise an exception if a requested state transition is illegal.

    This is a catch-all exception for model transitions. If it is called regularly a more specific
    exception should be defined.

    Parameters
    -------
    model_name : str
        The name of the model.
    current_state : enums.TableStatus
        The current state of the model.
    requested_state : enums.TableStatus
        The illegal state transition that was requested for the model.
    """

    def __init__(
        self,
        model_name: str,
        current_state: enums.TableStatus,
        requested_state: enums.TableStatus,
    ):
        super().__init__(
            f"Model `{model_name}` attempted an illegal transition from `{current_state}` to `{requested_state}`."
        )


""" Datum """


class DatumDoesNotExistError(Exception):
    """
    Raises an exception if the user tries to manipulate a datum that doesn't exist.

    Parameters
    -------
    uid : str
        The UID of the datum.
    """

    def __init__(self, uid: str):
        super().__init__(f"Datum with uid `{uid}` does not exist.")


class DatumAlreadyExistsError(Exception):
    """
    Raises an exception if the user tries to create a datum that already exists.

    Parameters
    -------
    uid : str
        The UID of the datum.
    """

    def __init__(self, uid: str):
        super().__init__(f"Datum with uid: `{uid}` already exists.")


class DatumsAlreadyExistError(Exception):
    """
    Raises an exception if the user tries to create a datum that already exists.

    Parameters
    -------
    uids
        The UIDs of the datums.
    """

    def __init__(self, uids: list[str]):
        super().__init__(f"Datums with uids: `{uids}` already exist.")


""" Annotation """


class AnnotationAlreadyExistsError(Exception):
    """
    Raises an exception if the user tries to create a annotation for a datum that already has annotation(s).

    Parameters
    -------
    datum_uid : str
        The UID of the datum.
    """

    def __init__(self, datum_uid: str):
        super().__init__(
            f"Annotation(s) for datum with uid: `{datum_uid}` already exist."
        )


class PredictionAlreadyExistsError(Exception):
    """
    Raises an exception if a prediction is duplicated.
    """

    def __init__(self):
        super().__init__(
            "A prediction with the same label already exists for this datum."
        )


class PredictionDoesNotExistError(Exception):
    """
    Raises an exception if a prediction does not exist for a given model, dataset, and datum
    """

    def __init__(self, model_name: str, dataset_name: str, datum_uid: str):
        super().__init__(
            f"A prediction for model `{model_name}` on dataset `{dataset_name}` and datum `{datum_uid}` does not exist."
        )


""" Evaluation """


class EvaluationDoesNotExistError(Exception):
    """
    Raises an exception if the user tries to manipulate an evaluation that doesn't exist.
    """

    def __init__(self):
        super().__init__("Evaluation does not exist.")


class EvaluationAlreadyExistsError(Exception):
    """
    Raises an exception if the user tries to create an evaluation that already exists.
    """

    def __init__(self):
        super().__init__("Evaluation with provided settings already exists.")


class EvaluationRunningError(Exception):
    """
    Raises an exception if the user tries to modify a dataset or model while an evaluation is running.
    """

    def __init__(
        self, dataset_name: str | None = None, model_name: str | None = None
    ):
        if dataset_name and model_name:
            msg = f"User action on model `{model_name}` and dataset `{dataset_name}` is blocked by at least one running evaluation."
        elif dataset_name:
            msg = f"User action on dataset `{dataset_name}` is blocked by at least one running evaluation."
        elif model_name:
            msg = f"User action on model `{model_name}` is blocked by at least one running evaluation."
        else:
            msg = "User action is blocked by at least one running evaluation."
        super().__init__(msg)


class EvaluationRequestError(Exception):
    """
    Raises an exception if the user request fails.
    """

    def __init__(self, msg: str, errors: list[Exception] | None = None):
        request_error = {
            "description": msg,
            "errors": [],
        }
        if errors is not None:
            request_error["errors"] = [
                {"name": type(error).__name__, "detail": str(error)}
                for error in errors
            ]
        super().__init__(json.dumps(request_error))


class EvaluationStateError(Exception):
    """
    Raises an exception if a requested state transition is illegal.

    This is a catch-all exception for evaluation transitions. If it is called regularly a more specific
    exception should be defined.

    Parameters
    -------
    evaluation_id : int
        The ID of the evaluation.
    current_state : enums.EvaluationStatus
        The current state of the evaluation.
    requested_state : enums.EvaluationStatus
        The illegal state transition that was requested for the evaluation.
    """

    def __init__(
        self,
        evaluation_id: int,
        current_state: enums.EvaluationStatus,
        requested_state: enums.EvaluationStatus,
    ):
        super().__init__(
            f"Evaluation `{evaluation_id}` attempted an illegal transition from `{current_state}` to `{requested_state}`."
        )


class InvalidLLMResponseError(Exception):
    """
    Raised when the response from the LLM is invalid for a given metric computation.
    """

    pass


class BadTestLLMClientsValueError(Exception):
    """
    Raised when a mock function in test_llm_clients.py receives a bad value.
    """

    pass


error_to_status_code = {
    # 400
    Exception: 400,
    ValueError: 400,
    AttributeError: 400,
    EvaluationRequestError: 400,
    # 404
    DatasetDoesNotExistError: 404,
    DatumDoesNotExistError: 404,
    ModelDoesNotExistError: 404,
    EvaluationDoesNotExistError: 404,
    PredictionDoesNotExistError: 404,
    # 409
    DatasetEmptyError: 409,
    DatasetAlreadyExistsError: 409,
    DatasetFinalizedError: 409,
    DatasetNotFinalizedError: 409,
    DatasetStateError: 409,
    ModelAlreadyExistsError: 409,
    ModelFinalizedError: 409,
    ModelNotFinalizedError: 409,
    ModelStateError: 409,
    DatumAlreadyExistsError: 409,
    AnnotationAlreadyExistsError: 409,
    PredictionAlreadyExistsError: 409,
    EvaluationAlreadyExistsError: 409,
    EvaluationRunningError: 409,
    EvaluationStateError: 409,
    # 500
    NotImplementedError: 500,
    # 503
    ServiceUnavailable: 503,
}


def create_http_error(
    error: (
        Exception
        | ValueError
        | AttributeError
        | DatasetDoesNotExistError
        | DatasetEmptyError
        | DatumDoesNotExistError
        | ModelDoesNotExistError
        | EvaluationDoesNotExistError
        | DatasetAlreadyExistsError
        | DatasetFinalizedError
        | DatasetNotFinalizedError
        | DatasetStateError
        | ModelAlreadyExistsError
        | ModelFinalizedError
        | ModelNotFinalizedError
        | ModelStateError
        | DatumAlreadyExistsError
        | AnnotationAlreadyExistsError
        | PredictionAlreadyExistsError
        | EvaluationAlreadyExistsError
        | EvaluationRunningError
        | EvaluationRequestError
        | EvaluationStateError
        | NotImplementedError
        | ServiceUnavailable
    ),
) -> HTTPException:
    """
    Creates a HTTP execption using a caught exception.

    The HTTPException is populated with the name and details of the caught exception.

    Parameters
    ----------
    error : Exception
        The exception that was caught and needs conversion.

    Returns
    -------
    fastapi.HTTPException
    """
    if type(error) in error_to_status_code:
        status_code = error_to_status_code[type(error)]
    else:
        status_code = 500
        logger.debug(
            f"`{type(error).__name__}` does not have a status_code assigned to it."
        )

    return HTTPException(
        status_code=status_code,
        detail=json.dumps(
            {
                "name": str(type(error).__name__),
                "detail": str(error),
                "timestamp": datetime.utcnow().timestamp(),
            }
        ),
    )

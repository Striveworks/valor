import json

from requests import Response, exceptions


class ClientException(Exception):
    def __init__(self, resp):
        self.status_code = resp.status_code
        self.detail = resp.json()["detail"]
        super().__init__(str(self.detail))


class ClientAlreadyConnectedError(Exception):
    def __init__(self):
        super().__init__("Client already connected.")


class ClientNotConnectedError(Exception):
    def __init__(self):
        super().__init__("Client not connected.")


class ClientConnectionFailed(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class ServiceUnavailable(ClientException):
    """
    Raises an exception if the Valor service is unavailble.
    """

    pass


class DatasetAlreadyExistsError(ClientException):
    """
    Raises an exception if the user tries to create a dataset with a name that already exists.
    """

    pass


class DatasetDoesNotExistError(ClientException):
    """
    Raises an exception if the user tries to manipulate a dataset that doesn't exist.
    """

    pass


class DatasetFinalizedError(ClientException):
    """
    Raises an exception if the user tries to add groundtruths to a dataset that has already been finalized.
    """

    pass


class DatasetNotFinalizedError(ClientException):
    """
    Raises an exception if the user tries to process a dataset that hasn't been finalized.
    """

    pass


class ModelAlreadyExistsError(ClientException):
    """
    Raises an exception if the user tries to create a model using a name that already exists in the database.
    """

    pass


class ModelDoesNotExistError(ClientException):
    """
    Raises an exception if the user tries to manipulate a model that doesn't exist.
    """

    pass


class ModelFinalizedError(ClientException):
    """
    Raises an exception if the user tries to add predictions to a model that has been finalized.
    """

    pass


class ModelNotFinalizedError(ClientException):
    """
    Raises an exception if the user tries to manipulate a model that hasn't been finalized.
    """

    pass


class DatumDoesNotExistError(ClientException):
    """
    Raises an exception if the user tries to manipulate a datum that doesn't exist.
    """

    pass


class DatumAlreadyExistsError(ClientException):
    """
    Raises an exception if the user tries to create a datum that already exists.

    """

    pass


class DatumsAlreadyExistsError(ClientException):
    """
    Raises an exception if the user tries to create a datum that already exists.

    """

    pass


class AnnotationAlreadyExistsError(ClientException):
    """
    Raises an exception if the user tries to create a annotation for a datum that already has annotation(s).
    """

    pass


class PredictionDoesNotExistError(ClientException):
    """
    Raises an exception if a prediction does not exist for a given model, dataset, and datum
    """

    pass


def raise_client_exception(resp: Response):
    try:
        resp_json = resp.json()
        try:
            error_dict = json.loads(resp_json["detail"])
            cls_name = error_dict["name"]
            if cls_name in globals() and issubclass(
                globals()[cls_name], ClientException
            ):
                raise globals()[cls_name](resp)
            else:
                raise ClientException(resp)
        except (TypeError, json.JSONDecodeError):
            raise ClientException(resp)
    except (exceptions.JSONDecodeError, KeyError):
        resp.raise_for_status()

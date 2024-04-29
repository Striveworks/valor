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
    pass


class DatasetAlreadyExistsError(ClientException):
    pass


class DatasetDoesNotExistError(ClientException):
    pass


class DatasetFinalizedError(ClientException):
    pass


class DatasetNotFinalizedError(ClientException):
    pass


class ModelAlreadyExistsError(ClientException):
    pass


class ModelDoesNotExistError(ClientException):
    pass


class ModelFinalizedError(ClientException):
    pass


class ModelNotFinalizedError(ClientException):
    pass


class ModelInferencesDoNotExist(ClientException):
    pass


class DatumDoesNotExistError(ClientException):
    pass


class DatumAlreadyExistsError(ClientException):
    pass


class AnnotationAlreadyExistsError(ClientException):
    pass


class PredictionDoesNotExistError(ClientException):
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

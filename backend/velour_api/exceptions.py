""" Dataset """


class DatasetAlreadyExistsError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Dataset with name `{name}` already exists.")


class DatasetDoesNotExistError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Dataset with name `{name}` does not exist.")


class DatasetFinalizedError(Exception):
    def __init__(self, name: str):
        return super().__init__(
            f"cannot edit dataset `{name}` since it has been finalized."
        )


class DatasetNotFinalizedError(Exception):
    def __init__(self, name: str):
        return super().__init__(
            f"cannot evaluate dataset `{name}` since it has not been finalized."
        )


""" Model """


class ModelAlreadyExistsError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name `{name}` already exists.")


class ModelDoesNotExistError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name `{name}` does not exist.")


class ModelFinalizedError(Exception):
    def __init__(self, *, dataset_name: str, model_name: str):
        return super().__init__(
            f"cannot edit inferences for model`{model_name}` on dataset `{dataset_name}` since it has been finalized"
        )


class ModelNotFinalizedError(Exception):
    def __init__(self, *, dataset_name: str, model_name: str):
        return super().__init__(
            f"cannot evaluate inferences for model `{model_name}` on dataset `{dataset_name}` since it has NOT been finalized."
        )


""" Datum """


class DatumDoesNotExistError(Exception):
    def __init__(self, uid: str):
        return super().__init__(f"Datum with uid `{uid}` does not exist.")


class DatumAlreadyExistsError(Exception):
    def __init__(self, uid: str):
        return super().__init__(f"Datum with uid: `{uid}` already exists.")


class DatumDoesNotBelongToDatasetError(Exception):
    def __init__(self, dataset_name: str, datum_uid: str):
        return super().__init__(
            f"Datum with uid: `{datum_uid}` does not belong to dataset `{dataset_name}`."
        )


""" Misc. """


class GroundTruthAlreadyExistsError(Exception):
    pass


class PredictionAlreadyExistsError(Exception):
    pass


class AnnotationAlreadyExistsError(Exception):
    pass


class MetaDatumAlreadyExistsError(Exception):
    def __init__(self, key: str):
        return super().__init__(f"Metadatum with key `{key}` already exists.")


""" Jobs & Stateflow"""


class StateflowError(Exception):
    def __init__(self, msg: str):
        return super().__init__(msg)


class JobDoesNotExistError(Exception):
    def __init__(self, id: str):
        return super().__init__(f"job with id `{id}` does not exist")


class JobStateError(Exception):
    def __init__(self, id: int, msg: str = "none"):
        return super().__init__(f"state error with job id: `{id}`, msg: {msg}")


class EvaluationJobDoesNotExistError(Exception):
    def __init__(self, id: str):
        return super().__init__(
            f"Evaluation job with id `{id}` does not exist"
        )

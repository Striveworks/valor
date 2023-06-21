from velour_api.enums import TableStatus


class DatasetAlreadyExistsError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Dataset with name '{name}' already exists.")


class ModelAlreadyExistsError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name '{name}' already exists.")


class DatasetDoesNotExistError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Dataset with name '{name}' does not exist.")


class DatasetIsFinalizedError(Exception):
    def __init__(self, name: str):
        return super().__init__(
            f"Cannot add images or annotations to dataset '{name}' since it is finalized."
        )


class DatasetIsNotFinalizedError(Exception):
    def __init__(self, name: str):
        return super().__init__(
            f"Cannot evaluate against dataset '{name}' since it is not finalized."
        )


class ModelDoesNotExistError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name '{name}' does not exist.")


class ModelIsFinalizedError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name '{name}' is not editable.")


class ModelIsNotFinalizedError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name '{name}' is not finalized.")


class ImageDoesNotExistError(Exception):
    def __init__(self, uid: str, dataset_name: str):
        return super().__init__(
            f"Image with uid '{uid}' does not exist in dataset '{dataset_name}'."
        )


class JobDoesNotExistError(Exception):
    def __init__(self, uid: str):
        return super().__init__(f"Job with uid '{uid} does not exist")


class InferencesAreNotFinalizedError(Exception):
    def __init__(self, dataset_name: str, model_name: str):
        return super().__init__(
            f"Inferences for model {model_name} on dataset {dataset_name} are not finalized."
        )


class InvalidStateTransitionError(Exception):
    def __init__(self, current: TableStatus, attempted: TableStatus):
        return super().__init__(
            f"Requested blocked state transition: {current} =/=> {attempted}"
        )


class InferenceAlreadyExistsError(Exception):
    def __init__(self, dataset_name: str, model_name: str):
        return super().__init__(
            f"Inference on dataset {dataset_name} from model '{model_name}' already exists."
        )


class InferenceDoesNotExistError(Exception):
    def __init__(self, dataset_name: str, model_name: str):
        return super().__init__(
            f"Inference on dataset {dataset_name} from model '{model_name}' does not exist."
        )


class InvalidStateError(Exception):
    def __init__(self, msg: str):
        return super().__init__(msg)

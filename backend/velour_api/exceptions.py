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


class DatasetIsDraftError(Exception):
    def __init__(self, name: str):
        return super().__init__(
            f"Cannot evaluate against dataset '{name}' since it is still a draft."
        )


class ModelDoesNotExistError(Exception):
    def __init__(self, name: str):
        return super().__init__(f"Model with name '{name}' does not exist.")


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

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


class DatasetStateFlowError(Exception):
    def __init__(
        self,
        dataset_name: str,
        current: TableStatus = None,
    ):
        # Current event
        if current == TableStatus.CREATE:
            errmsg = f"Cannot perform any action on dataset '{dataset_name}' as it has not been finalized."
        elif current == TableStatus.READY:
            errmsg = f"Dataset '{dataset_name}' cannot be modified as it has been finalized."
        elif current == TableStatus.EVALUATE:
            errmsg = f"Cannot perform any action on dataset '{dataset_name}' until all models are finalized and/or evaluations are finished."
        elif current == TableStatus.DELETE:
            errmsg = f"Dataset '{dataset_name}' is currently being deleted."
        # Unknown event
        else:
            errmsg = f"{dataset_name} does not exist."
        return super().__init__(errmsg)


class ModelStateFlowError(Exception):
    def __init__(
        self,
        model_name: str,
        current: TableStatus = None,
    ):
        # Current event
        if current == TableStatus.CREATE:
            errmsg = f"Cannot perform any action on model '{model_name}' as it has not been finalized."
        elif current == TableStatus.READY:
            errmsg = f"Model '{model_name}' cannot be modified as it has been finalized."
        elif current == TableStatus.EVALUATE:
            errmsg = f"Cannot perform any action on model '{model_name}' until evaluations are finished."
        elif current == TableStatus.DELETE:
            errmsg = f"Model '{model_name}' is currently being deleted."
        # Unknown event
        else:
            errmsg = f"{model_name} does not exist."
        return super().__init__(errmsg)

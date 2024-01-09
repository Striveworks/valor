from velour_api import enums


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


class DatasetIsEmptyError(Exception):
    """
    Raises an exception if the user tries to manipulate an empty dataset.

    Parameters
    -------
    name : str
        The name of the dataset.
    """

    def __init__(self, name: str):
        super().__init__(
            f"Dataset with name `{name}` contains no groundtruths."
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

    def __init__(self, name: str):
        super().__init__(
            f"cannot evaluate dataset `{name}` since it has not been finalized."
        )


class DatasetStateError(Exception):
    """
    Raise an exception if a requested state transition is illegal.

    This is a catch-all exception, if it is called regularly a more specific
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


class ModelIsEmptyError(Exception):
    """
    Raises an exception if the user tries to manipulate an empty model.

    Parameters
    -------
    name : str
        The name of the model.
    """

    def __init__(self, name: str):
        super().__init__(f"Model with name `{name}` contains no inferences.")


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


class ModelInferencesDoNotExist(Exception):
    def __init__(self, *, dataset_name: str, model_name: str):
        """
        Raises an exception if the user tries to manipulate an inference that doesn't exist.

        Parameters
        -------
        dataset_name : str
            The name of the dataset.
        model_name : str
            The name of the model.
        """

        super().__init__(
            f"inferences for model `{model_name}` over dataset `{dataset_name}` do not exist."
        )


class ModelStateError(Exception):
    """
    Raise an exception if a requested state transition is illegal.

    This is a catch-all exception, if it is called regularly a more specific
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
        The uid of the datum.
    """

    def __init__(self, uid: str):
        super().__init__(f"Datum with uid `{uid}` does not exist.")


class DatumAlreadyExistsError(Exception):
    """
    Raises an exception if the user tries to create a datum that already exists.

    Parameters
    -------
    uid : str
        The uid of the datum.
    """

    def __init__(self, uid: str):
        super().__init__(f"Datum with uid: `{uid}` already exists.")


class DatumDoesNotBelongToDatasetError(Exception):
    """
    Raises an exception if the user tries to manipulate a datum that doesn't exist on a particular dataset.

    Parameters
    -------
    dataset_name : str
        The name of the dataset.
    datum_uid : str
        The uid of the datum.
    """

    def __init__(self, dataset_name: str, datum_uid: str):
        super().__init__(
            f"Datum with uid: `{datum_uid}` does not belong to dataset `{dataset_name}`."
        )


""" Annotation """


class AnnotationAlreadyExistsError(Exception):
    """
    Raises an exception if the user tries to create a annotation for a datum that already has annotation(s).

    Parameters
    -------
    datum_uid : str
        The uid of the datum.
    """

    def __init__(self, datum_uid: str):
        super().__init__(
            f"Annotation(s) for datum with uid: `{datum_uid}` already exist."
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


class EvaluationStateError(Exception):
    """
    Raises an exception if a requested state transition is illegal.

    This is a catch-all exception, if it is called regularly a more specific
    exception should be defined.

    Parameters
    -------
    evaluation_id : int
        The id of the evaluation.
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

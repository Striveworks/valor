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
        super().__init__(
            f"Model with name `{name}` contains no inferences."
        )


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
            f"cannot edit inferences for model`{model_name}` on dataset `{dataset_name}` since it has been finalized"
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


""" Misc. """


class GroundTruthAlreadyExistsError(Exception):
    """Raises an exception if the user tries to create a `GroundTruth` that already exists."""
    pass


class AnnotationAlreadyExistsError(Exception):
    """Raises an exception if the user tries to create an `Annotation` that already exists."""
    pass


""" Jobs """


class JobDoesNotExistError(Exception):
    """
    Raises an exception if the user queries a job ID which doesn't exist.

    Parameters
    -------
    id : str
        The id of the job.
    """

    def __init__(self, id: str):
        super().__init__(f"job with id `{id}` does not exist")


class JobStateError(Exception):
    """
    Raises an exception if the user tries to manipulate a job at the wrong time (e.g., deleting a job that is currently processing).

    Parameters
    -------
    id : str
        The id of the job.
    msg : str
        The message of the stateflow error.
    """

    def __init__(self, id: str, msg: str = ""):
        super().__init__(f"state error with job id: `{id}`, msg: {msg}")

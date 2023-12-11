from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models


def get_model(
    db: Session,
    name: str,
) -> models.Model:
    """
    Fetch a model from the database.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    name : str
        The name of the model.

    Returns
    ----------
    models.Model
        The requested model.

    """

    model = (
        db.query(models.Model).where(models.Model.name == name).one_or_none()
    )
    if model is None:
        raise exceptions.ModelDoesNotExistError(name)
    return model

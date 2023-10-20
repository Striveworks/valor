from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models


def get_model(
    db: Session,
    name: str,
) -> models.Model:
    model = (
        db.query(models.Model).where(models.Model.name == name).one_or_none()
    )
    if model is None:
        raise exceptions.ModelDoesNotExistError(name)
    return model

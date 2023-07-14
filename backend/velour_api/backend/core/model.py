from sqlalchemy.orm import Session

from velour_api.backend import models


def get_model(
    db: Session,
    name: str,
) -> models.Model:
    return (
        db.query(models.Model).where(models.Model.name == name).one_or_none()
    )

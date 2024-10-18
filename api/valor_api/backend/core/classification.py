from sqlalchemy import ScalarSelect, and_, delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import models

from valor_lite.classification import DataLoader


def insert_classifications(
    dataset_id: int,
    model_id: int,
    datum_id: int,
    classifications: list[schemas.Classification],
):
    loader = DataLoader()
    loader.add_data(classifications)
    loader.finalize()


def load_cache(
    db: Session,
    dataset_id: int,
    model_id: int,
):
    return db.query(models.Classification).where(
        and_(
            models.Classification.dataset_id == dataset_id,
            models.Classification.model_id == model_id,
        )
    ).all()


def load_cache_from_filter(
    db: Session,
    filter_: schemas.Filter,
):
    pass


def delete_classifications(
    dataset_id: int | None = None,
    model_id: int | None = None,
):
    pass

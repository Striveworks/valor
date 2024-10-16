from sqlalchemy import ScalarSelect, and_, delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import models


def load_cache(
    dataset_id: int,
    model_id: int,
):
    pass


def load_cache_from_filter(filter_):
    pass


def insert_classifications(
    dataset_id: int,
    model_id: int,
    classifications: list[schemas.Classification],
):
    pass


def delete_classifications(
    dataset_id: int | None = None,
    model_id: int | None = None,
):
    pass

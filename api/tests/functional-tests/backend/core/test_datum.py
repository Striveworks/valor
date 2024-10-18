import pytest
from sqlalchemy import func, select, insert
from sqlalchemy.orm import Session

from valor_api import exceptions, schemas

from valor_api.backend.core.datum import insert_datums, delete_datums


@pytest.fixture
def datums() -> list[schemas.Classification | schemas.ObjectDetection | schemas.SemanticSegmentation]:
    return [
        schemas.Classification(
            uid="uid0",
            groundtruth=
        )
    ]



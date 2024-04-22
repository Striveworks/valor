import pytest
from sqlalchemy.orm import Session

from valor import Annotation, Dataset, Datum, Label, Model
from valor.enums import TaskType


def test_concurrent_label_creation(db: Session):
    pass

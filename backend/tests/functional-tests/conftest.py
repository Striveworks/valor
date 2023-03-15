import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from velour_api import models, schemas
from velour_api.database import Base, create_db, make_session

# get all velour table names
classes = [
    v
    for v in models.__dict__.values()
    if isinstance(v, type) and issubclass(v, Base)
]
tablenames = [v.__tablename__ for v in classes if hasattr(v, "__tablename__")]


def drop_all(db):
    db.execute(text(f"DROP TABLE {', '.join(tablenames)};"))
    db.commit()


def random_mask_bytes(size: tuple[int, int]) -> bytes:
    mask = np.random.randint(0, 2, size=size, dtype=bool)
    mask = Image.fromarray(mask)
    f = io.BytesIO()
    mask.save(f, format="PNG")
    f.seek(0)
    return f.read()


@pytest.fixture
def mask_bytes1():
    return random_mask_bytes(size=(32, 64))


@pytest.fixture
def mask_bytes2():
    return random_mask_bytes(size=(16, 12))


@pytest.fixture
def img1() -> schemas.Image:
    return schemas.Image(uid="uid1", height=1000, width=2000)


@pytest.fixture
def img2() -> schemas.Image:
    return schemas.Image(uid="uid2", height=1600, width=1200)


@pytest.fixture
def db():
    """This fixture provides a db session. a `RuntimeError` is raised if
    a velour tablename already exists. At teardown, all velour tables are wiped.
    """
    db = make_session()
    inspector = inspect(db.connection())
    for tablename in tablenames:
        if inspector.has_table(tablename):
            raise RuntimeError(
                f"Table {tablename} already exists; "
                "functional tests should be run with an empty db."
            )

    create_db()
    yield db
    # teardown
    drop_all(db)


@pytest.fixture
def dset(db: Session) -> models.Dataset:
    dset = models.Dataset(name="dset")
    db.add(dset)
    db.commit()

    return dset


@pytest.fixture
def img(db: Session, dset: models.Dataset) -> models.Image:
    img = models.Image(uid="uid", dataset_id=dset.id, height=1000, width=2000)
    db.add(img)
    db.commit()

    return img

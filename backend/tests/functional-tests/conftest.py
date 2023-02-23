import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import inspect, text

from velour_api import models
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

    create_db(timeout=30)
    yield db
    # teardown
    drop_all(db)

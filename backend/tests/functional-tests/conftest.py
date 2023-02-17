import pytest
from sqlalchemy import inspect, text

from velour_api import models, schemas
from velour_api.database import Base, SessionLocal, create_db

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


@pytest.fixture
def db():
    """This fixture provides a db session. a `RuntimeError` is raised if
    a velour tablename already exists. At teardown, all velour tables are wiped.
    """
    db = SessionLocal()
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


@pytest.fixture
def poly_without_hole() -> schemas.PolygonWithHole:
    # should have area 45.5
    return schemas.PolygonWithHole(polygon=[(4, 10), (9, 7), (11, 2), (2, 2)])


@pytest.fixture
def poly_with_hole() -> schemas.PolygonWithHole:
    # should have area 100 - 8 = 92
    return schemas.PolygonWithHole(
        polygon=[(0, 10), (10, 10), (10, 0), (0, 0)],
        hole=[(2, 4), (2, 8), (6, 4)],
    )

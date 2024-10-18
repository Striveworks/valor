import pytest
import datetime

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import models
from valor_api.backend.core.metadata import insert_metadata, delete_metadata


@pytest.fixture
def metadata() -> schemas.Metadata:
    return schemas.Metadata(
        string={
            "key1": "hello world",
            "key2": "foobar",
        },
        integer={
            "key3": 123,
            "key4": 321,
        },
        floating={
            "key5": 1.23,
            "key6": 3.21,
        },
        geospatial={
            "key7": schemas.GeoJSON(
                type="Point", 
                coordinates=[10, 15]
            ),
            "key8": schemas.GeoJSON(
                type="Polygon",
                coordinates=[
                    [
                        [0, 0],
                        [0, 10],
                        [10, 0],
                        [0, 0]
                    ]
                ]
            ),
        },
        datetime={
            "key9": datetime.datetime(
                year=2024,
                month=1,
                day=1,
                hour=1,
                minute=1,
                second=1,
            ),
            "key10": datetime.datetime(
                year=2024,
                month=12,
                day=12,
                hour=12,
                minute=12,
                second=12,
            ),
        },
        date={
            "key11": datetime.date(
                year=2024,
                month=1,
                day=1,
            ),
            "key12": datetime.date(
                year=2024,
                month=12,
                day=12,
            ),
        },
        time={
            "key13": datetime.time(
                hour=1,
                minute=1,
                second=1,
            ),
            "key14": datetime.time(
                hour=12,
                minute=12,
                second=12,
            ),
        },
    )


def test_crud_metadata(db: Session, metadata: schemas.Metadata):

    # edge case - metadata does not exist
    assert insert_metadata(db=db, metadata=None) is None

    # test metadata creation
    insert_metadata(db=db, metadata=metadata)
    rows = db.query(models.Metadata).all()
    assert len(rows) == 1
    
    # extract metadata id
    metadata_id = rows[0].id

    # test string metadata
    rows = db.query(models.String).all()
    assert len(rows) == 2

    assert rows[0].metadata_id == metadata_id
    assert rows[0].key == "key1"
    assert rows[0].value == "hello world"

    assert rows[1].metadata_id == metadata_id
    assert rows[1].key == "key2"
    assert rows[1].value == "foobar"

    # test integer metadata
    rows = db.query(models.Integer).all()
    assert len(rows) == 2

    assert rows[0].metadata_id == metadata_id
    assert rows[0].key == "key3"
    assert rows[0].value == 123

    assert rows[1].metadata_id == metadata_id
    assert rows[1].key == "key4"
    assert rows[1].value == 321

    # test float metadata
    rows = db.query(models.Float).all()
    assert len(rows) == 2

    assert rows[0].metadata_id == metadata_id
    assert rows[0].key == "key5"
    assert rows[0].value == 1.23

    assert rows[1].metadata_id == metadata_id
    assert rows[1].key == "key6"
    assert rows[1].value == 3.21

    # test geospatial metadata
    rows = db.query(
        select(
            models.Geospatial.metadata_id,
            models.Geospatial.key,
            func.ST_AsGeoJSON(models.Geospatial.value)
        ).subquery()
    ).all()
    assert len(rows) == 2

    assert rows[0][0] == metadata_id
    assert rows[0][1] == "key7"
    assert rows[0][2] == '{"type":"Point","coordinates":[10,15]}'

    assert rows[1][0] == metadata_id
    assert rows[1][1] == "key8"
    assert rows[1][2] == '{"type":"Polygon","coordinates":[[[0,0],[0,10],[10,0],[0,0]]]}'

    # test datetime metadata
    rows = db.query(models.DateTime).all()
    assert len(rows) == 2

    assert rows[0].metadata_id == metadata_id
    assert rows[0].key == "key9"
    assert rows[0].value == datetime.datetime(2024, 1, 1, 1, 1, 1)

    assert rows[1].metadata_id == metadata_id
    assert rows[1].key == "key10"
    assert rows[1].value == datetime.datetime(2024, 12, 12, 12, 12, 12)

    # test date metadata
    rows = db.query(models.Date).all()
    assert len(rows) == 2

    assert rows[0].metadata_id == metadata_id
    assert rows[0].key == "key11"
    assert rows[0].value == datetime.date(2024, 1, 1)

    assert rows[1].metadata_id == metadata_id
    assert rows[1].key == "key12"
    assert rows[1].value == datetime.date(2024, 12, 12)

    # test time metadata
    rows = db.query(models.Time).all()
    assert len(rows) == 2

    assert rows[0].metadata_id == metadata_id
    assert rows[0].key == "key13"
    assert rows[0].value == datetime.time(1, 1, 1)

    assert rows[1].metadata_id == metadata_id
    assert rows[1].key == "key14"
    assert rows[1].value == datetime.time(12, 12, 12)
    
    # test metadata deletion
    delete_metadata(db=db, metadata_id=metadata_id)

    assert len(db.query(models.Metadata).all()) == 0
    assert len(db.query(models.String).all()) == 0
    assert len(db.query(models.Integer).all()) == 0
    assert len(db.query(models.Float).all()) == 0
    assert len(db.query(models.Geospatial).all()) == 0
    assert len(db.query(models.DateTime).all()) == 0
    assert len(db.query(models.Date).all()) == 0
    assert len(db.query(models.Time).all()) == 0
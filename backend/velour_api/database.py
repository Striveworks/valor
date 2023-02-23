import logging
import os
import time

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import text

SQLALCHEMY_DATABASE_URL = f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/postgres"

logging.debug(f"SQLALCHEMY_DATABASE_URL: {SQLALCHEMY_DATABASE_URL}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
make_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def make_session() -> Session:
    """Creates a session and enables the gdal drivers (needed for raster support)"""
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    db.execute(text("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';"))

    return db


Base = declarative_base()


def create_db(timeout: int = 15):
    from . import models

    db = make_session()
    start_time = time.time()
    while True:
        try:
            db.execute(
                text("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';")
            )
            db.commit()
            # create raster extension if it doesn't exist
            if (
                db.execute(
                    text(
                        "SELECT * FROM pg_extension WHERE extname='postgis_raster';"
                    )
                ).scalar()
                is None
            ):
                db.execute(text("CREATE EXTENSION postgis_raster;"))
                db.commit()
            db.close()
            models.Base.metadata.create_all(bind=engine)
            break
        except (OperationalError, ProgrammingError) as e:
            if time.time() - start_time >= timeout:
                raise RuntimeError(
                    f"Failed to connect to database within {timeout} seconds. Error: {str(e)}"
                )
            time.sleep(2)

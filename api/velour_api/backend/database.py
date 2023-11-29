import os
import time

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import text

from velour_api import logger
from velour_api.backend import models

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)

TIMEOUT = 30

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

logger.debug(
    f"POSTGRES_HOST: {POSTGRES_HOST}:{POSTGRES_PORT}, POSTGRES_USERNAME: {POSTGRES_USERNAME}, "
    f"POSTGRES_PASSWORD: {'null' if POSTGRES_PASSWORD is None else 'not null'}, POSTGRES_DB: {POSTGRES_DB} "
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
make_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def retry_connection(f):

    global TIMEOUT

    def wrapper(*args, **kwargs):
        start_time = time.time()
        while True:
            try:
                return f(*args, **kwargs)
            except (
                psycopg2.OperationalError,
                OperationalError,
                ProgrammingError,
            ) as e:
                if time.time() - start_time >= TIMEOUT:
                    raise RuntimeError(
                        f"Method {f.__name__} failed to connect to database within {TIMEOUT} seconds, with error: {str(e)}"
                    )
            time.sleep(2)

    return wrapper


@retry_connection
def make_session() -> Session:
    """Creates a session and enables the gdal drivers (needed for raster support)"""
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    db.execute(text("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';"))
    db.commit()
    return db


Base = declarative_base()


@retry_connection
def create_db():
    db = make_session()
    # create postgis and raster extensions if they don't exist
    for extension in ["postgis", "postgis_raster"]:
        if (
            db.execute(
                text(
                    f"SELECT * FROM pg_extension WHERE extname='{extension}';"
                )
            ).scalar()
            is None
        ):
            db.execute(text(f"CREATE EXTENSION {extension};"))
            db.commit()

    db.close()
    models.Base.metadata.create_all(bind=engine)

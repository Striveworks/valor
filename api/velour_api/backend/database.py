import os
import time

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import text

from velour_api import logger

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?application_name=velour"

logger.debug(
    f"POSTGRES_HOST: {POSTGRES_HOST}:{POSTGRES_PORT}, POSTGRES_USERNAME: {POSTGRES_USERNAME}, "
    f"POSTGRES_PASSWORD: {'null' if POSTGRES_PASSWORD is None else 'not null'}, POSTGRES_DB: {POSTGRES_DB} "
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)

check_db_connection_called = False


def check_db_connection(db: Session) -> None:
    """Check if the database connection is valid

    Raises
    ------
    RuntimeError
        If unable to connect to the database within 30 seconds
    """
    timeout = 30
    start_time = time.time()
    while True:
        try:
            db.execute(text("SELECT 1"))
            break
        except (
            psycopg2.OperationalError,
            OperationalError,
            ProgrammingError,
        ) as e:
            if time.time() - start_time >= timeout:
                raise RuntimeError(
                    f"Failed to connect to database within {timeout} seconds, with error: {str(e)}"
                )

    global check_db_connection_called
    check_db_connection_called = True


def make_session() -> Session:
    """Creates a session and enables the gdal drivers (needed for raster support). The first
    time this is called we verify that the we can actually connect to the database.
    """
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    if not check_db_connection_called:
        check_db_connection(db)
    return db


Base = declarative_base()

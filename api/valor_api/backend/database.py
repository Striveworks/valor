import os
import time
from typing import Callable

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import text

from valor_api import logger

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?application_name=valor"

logger.debug(
    f"POSTGRES_HOST: {POSTGRES_HOST}:{POSTGRES_PORT}, POSTGRES_USERNAME: {POSTGRES_USERNAME}, "
    f"POSTGRES_PASSWORD: {'null' if POSTGRES_PASSWORD is None else 'not null'}, POSTGRES_DB: {POSTGRES_DB} "
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)


def try_to_enable_gdal_drivers(db: Session) -> None:
    """Tries to enable the GDAL drivers for the database. However in some cases
    the application may not have permission to and so that must be taken care of
    out side of this application
    """
    try:
        # enable for future sessions
        db.execute(
            text(
                f"ALTER DATABASE {POSTGRES_DB} SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';"
            )
        )

        # enable for this session
        db.execute(text("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';"))
        db.commit()
    except (psycopg2.OperationalError, OperationalError, ProgrammingError):
        db.rollback()
        db.close()


def check_db_connection(db: Session, timeout: int = 30) -> None:
    """Check if the database connection is valid

    Parameters
    ----------
    db : Session
        The database connection
    timeout : int, optional
        The number of seconds to wait for the database to connect, by default 30

    Raises
    ------
    RuntimeError
        If unable to connect to the database within 30 seconds
    """
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


first_time_make_session_called = True


def make_make_session() -> Callable[[], Session]:

    first_time_make_session_called = True

    def make_session() -> Session:
        """Creates a session and enables the gdal drivers (needed for raster support). The first
        time this is called we verify that the we can actually connect to the database.
        """
        nonlocal first_time_make_session_called
        db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
        if first_time_make_session_called:
            check_db_connection(db)
            try_to_enable_gdal_drivers(db)
            first_time_make_session_called = False
        return db

    return make_session


make_session = make_make_session()

Base = declarative_base()

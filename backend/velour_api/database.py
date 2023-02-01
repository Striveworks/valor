import logging
import os
import time

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/postgres"

logging.debug(f"SQLALCHEMY_DATABASE_URL: {SQLALCHEMY_DATABASE_URL}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def create_db(timeout: int = 15):
    from . import models

    start_time = time.time()
    while True:
        try:
            models.Base.metadata.create_all(bind=engine)
            break
        except OperationalError:
            if time.time() - start_time >= timeout:
                raise RuntimeError(
                    f"Failed to connect to database within {timeout} seconds."
                )
            time.sleep(2)

import json
import os
import time

import redis

from velour_api import logger
from velour_api.schemas import Stateflow

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_SSL = bool(os.getenv("REDIS_SSL"))

TIMEOUT = 30

# global connection to redis
r: redis.Redis = None


def retry_connection(f):
    global TIMEOUT

    def wrapper(*args, **kwargs):
        start_time = time.time()
        while True:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if time.time() - start_time >= TIMEOUT:
                    logger.debug(
                        f"REDIS_HOST: {REDIS_HOST}, REDIS_PORT: {REDIS_PORT}, REDIS_DB: {REDIS_DB}, "
                        f"REDIS_PASSWORD: {'null' if REDIS_PASSWORD is None else 'not null'}, "
                        f"REDIS_USERNAME: {REDIS_USERNAME}, REDIS_SSL: {REDIS_SSL}"
                    )
                    raise RuntimeError(
                        f"Method {f.__name__} failed to connect to database within {TIMEOUT} seconds, with error: {str(e)}"
                    )
            time.sleep(2)

    return wrapper


@retry_connection
def connect_to_redis():
    global r
    if r is not None:
        return

    connection = redis.Redis(
        REDIS_HOST,
        db=REDIS_DB,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        ssl=REDIS_SSL,
    )
    connection.ping()  # this should fail if redis does not exist
    logger.info(
        f"succesfully connected to redis instance at {REDIS_HOST}:{REDIS_PORT}"
    )

    # Only set if connected
    r = connection


def needs_redis(fn):
    def wrapper(*args, **kwargs):
        if r is None:
            connect_to_redis()
        return fn(*args, **kwargs)

    return wrapper


""" Stateflow """


@needs_redis
def get_stateflow() -> Stateflow:
    json_str = r.get("stateflow")
    if json_str is None or not isinstance(json_str, bytes):
        return Stateflow()
    info = json.loads(json_str)
    return Stateflow(**info)


@needs_redis
def set_stateflow(stateflow: Stateflow):
    r.set("stateflow", stateflow.model_dump_json())

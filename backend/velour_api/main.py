import time

from fastapi import Depends, FastAPI
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

# wait up to 30 seconds for the db to spin up
timeout = 15
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


app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/groundtruth-detections")
def create_groundtruth_detections(
    detections: list[schemas.GroundTruthDetectionCreate],
    db: Session = Depends(get_db),
) -> list[int]:
    return crud.create_groundtruth_detections(db=db, detections=detections)


@app.post("/predicted-detections")
def create_predicted_detections(
    detections: list[schemas.PredictedDetectionCreate],
    db: Session = Depends(get_db),
) -> list[int]:
    return crud.create_predicted_detections(db=db, detections=detections)

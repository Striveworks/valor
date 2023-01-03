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


@app.post(
    "/groundtruth-detections", response_model=schemas.GroundTruthDetection
)
def create_groundtruth_detection(
    detection: schemas.GroundTruthDetectionCreate,
    db: Session = Depends(get_db),
) -> schemas.GroundTruthDetection:
    return crud.create_groundtruth_detection(db=db, detection=detection)


@app.post("/predicted-detections", response_model=schemas.GroundTruthDetection)
def create_predicted_detection(
    detection: schemas.PredictedDetectionCreate,
    db: Session = Depends(get_db),
) -> schemas.PredictedDetection:
    return crud.create_predicted_detection(db=db, detection=detection)

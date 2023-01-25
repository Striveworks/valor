from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from . import crud, schemas
from .database import SessionLocal, create_db

app = FastAPI()

create_db()


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

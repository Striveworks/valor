from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post(
    "/groundtruth-detections", response_model=schemas.GroundTruthDetection
)
def create_gt_detection(
    detection: schemas.GroundTruthDetectionCreate,
    db: Session = Depends(get_db),
) -> schemas.GroundTruthDetection:
    return crud.create_gt_detection(db=db, detection=detection)

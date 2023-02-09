from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from velour_api import crud, schemas
from velour_api.database import SessionLocal, create_db

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
    data: schemas.GroundTruthDetectionsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        return crud.create_groundtruth_detections(db=db, data=data)
    except crud.DatasetIsFinalizedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/predicted-detections")
def create_predicted_detections(
    detections: list[schemas.PredictedDetectionsCreate],
    db: Session = Depends(get_db),
) -> list[int]:
    return crud.create_predicted_detections(db=db, detections=detections)


@app.get("/datasets", status_code=200)
def get_datasets(db: Session = Depends(get_db)) -> list[schemas.Dataset]:
    return crud.get_datasets(db)


@app.post("/datasets", status_code=201)
def create_dataset(
    dataset: schemas.DatasetCreate, db: Session = Depends(get_db)
):
    try:
        crud.create_dataset(db=db, dataset=dataset)
    except crud.DatasetAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.put("/datasets/{dataset_name}")
def get_dataset(dataset_name: str) -> schemas.Dataset:
    # returns nothing for now, just verifies dataset exists
    dset = crud.get_dataset(name=dataset_name)
    return schemas.Dataset(name=dset.name, draft=dset.draft)


@app.put("/datasets/{dataset_name}/finalize", status_code=200)
def finalize_dataset(dataset_name: str, db: Session = Depends(get_db)):
    try:
        crud.finalize_dataset(db, dataset_name)
    except crud.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/datasets/{dataset_name}/labels", status_code=200)
def get_dataset_labels(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    try:
        labels = crud.get_labels_in_dataset(db, dataset_name)
    except crud.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [
        schemas.Label(key=label.key, value=label.value) for label in labels
    ]


@app.get("/datasets/{dataset_name}/images", status_code=200)
def get_dataset_images(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Image]:
    try:
        images = crud.get_images_in_dataset(db, dataset_name)
    except crud.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [schemas.Image(uri=image.uri) for image in images]


@app.delete("/datasets/{dataset_name}")
def delete_dataset(dataset_name: str, db: Session = Depends(get_db)) -> None:
    return crud.delete_dataset(db, dataset_name)


@app.get("/labels", status_code=200)
def get_labels(db: Session = Depends(get_db)) -> list[schemas.Label]:
    return crud.get_all_labels(db)

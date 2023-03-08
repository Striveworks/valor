import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from velour_api import auth, crud, exceptions, logger, schemas
from velour_api.database import create_db, make_session
from velour_api.settings import auth_settings

token_auth_scheme = auth.OptionalHTTPBearer()


app = FastAPI(root_path=os.getenv("API_ROOT_PATH", ""))

create_db()

logger.info(
    f"API started {'WITHOUT' if auth_settings.no_auth else 'WITH'} authentication"
)


def get_db():
    db = make_session()
    try:
        yield db
    finally:
        db.close()


# should move the following routes to be behind /datasets/{dataset}/ ?
@app.post("/groundtruth-detections", dependencies=[Depends(token_auth_scheme)])
def create_groundtruth_detections(
    data: schemas.GroundTruthDetectionsCreate, db: Session = Depends(get_db)
) -> list[int]:
    try:
        return crud.create_groundtruth_detections(db=db, data=data)
    except exceptions.DatasetIsFinalizedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/predicted-detections", dependencies=[Depends(token_auth_scheme)])
def create_predicted_detections(
    data: schemas.PredictedDetectionsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        return crud.create_predicted_detections(db=db, data=data)
    except exceptions.ImageDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post(
    "/groundtruth-segmentations", dependencies=[Depends(token_auth_scheme)]
)
def create_groundtruth_segmentations(
    data: schemas.GroundTruthSegmentationsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        logger.debug(f"got: {data}")
        return crud.create_groundtruth_segmentations(db=db, data=data)
    except exceptions.DatasetIsFinalizedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post(
    "/predicted-segmentations", dependencies=[Depends(token_auth_scheme)]
)
def create_predicted_segmentations(
    data: schemas.PredictedSegmentationsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        return crud.create_predicted_segmentations(db=db, data=data)
    except exceptions.ImageDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post(
    "/groundtruth-classifications", dependencies=[Depends(token_auth_scheme)]
)
def create_groundtruth_classifications(
    data: schemas.GroundTruthImageClassificationsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        logger.debug(f"got: {data}")
        return crud.create_ground_truth_image_classifications(db=db, data=data)
    except exceptions.DatasetIsFinalizedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post(
    "/predicted-classifications", dependencies=[Depends(token_auth_scheme)]
)
def create_predicted_classifications(
    data: schemas.PredictedImageClassificationsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        return crud.create_predicted_image_classifications(db=db, data=data)
    except exceptions.ImageDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets", status_code=200, dependencies=[Depends(token_auth_scheme)]
)
def get_datasets(db: Session = Depends(get_db)) -> list[schemas.Dataset]:
    return crud.get_datasets(db)


@app.post(
    "/datasets", status_code=201, dependencies=[Depends(token_auth_scheme)]
)
def create_dataset(
    dataset: schemas.DatasetCreate, db: Session = Depends(get_db)
):
    try:
        crud.create_dataset(db=db, dataset=dataset)
    except exceptions.DatasetAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/datasets/{dataset_name}", dependencies=[Depends(token_auth_scheme)])
def get_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> schemas.Dataset:
    try:
        dset = crud.get_dataset(db, dataset_name=dataset_name)
        return schemas.Dataset(name=dset.name, draft=dset.draft)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/datasets/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def finalize_dataset(dataset_name: str, db: Session = Depends(get_db)):
    try:
        crud.finalize_dataset(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_labels(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    try:
        labels = crud.get_labels_in_dataset(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [
        schemas.Label(key=label.key, value=label.value) for label in labels
    ]


@app.get(
    "/datasets/{dataset_name}/images",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_images(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Image]:
    try:
        images = crud.get_images_in_dataset(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [
        schemas.Image(uri=image.uri, height=image.height, width=image.width)
        for image in images
    ]


@app.delete(
    "/datasets/{dataset_name}", dependencies=[Depends(token_auth_scheme)]
)
def delete_dataset(dataset_name: str, db: Session = Depends(get_db)) -> None:
    return crud.delete_dataset(db, dataset_name)


@app.get("/models", status_code=200, dependencies=[Depends(token_auth_scheme)])
def get_models(db: Session = Depends(get_db)) -> list[schemas.Model]:
    return crud.get_models(db)


@app.post(
    "/models", status_code=201, dependencies=[Depends(token_auth_scheme)]
)
def create_model(model: schemas.Model, db: Session = Depends(get_db)):
    try:
        crud.create_model(db=db, model=model)
    except exceptions.ModelAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/models/{model_name}", dependencies=[Depends(token_auth_scheme)])
def get_model(model_name: str) -> schemas.Model:
    try:
        dset = crud.get_model(name=model_name)
        return schemas.Model(name=dset.name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/models/{model_name}", dependencies=[Depends(token_auth_scheme)])
def delete_model(model_name: str, db: Session = Depends(get_db)) -> None:
    return crud.delete_model(db, model_name)


@app.get("/labels", status_code=200, dependencies=[Depends(token_auth_scheme)])
def get_labels(db: Session = Depends(get_db)) -> list[schemas.Label]:
    return crud.get_all_labels(db)


@app.get("/user")
def user(
    token: HTTPAuthorizationCredentials | None = Depends(token_auth_scheme),
) -> schemas.User:
    token_payload = auth.verify_token(token)
    return schemas.User(email=token_payload.get("email"))

import os

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from velour_api import auth, crud, enums, exceptions, jobs, logger, schemas
from velour_api.database import create_db, make_session
from velour_api.settings import auth_settings

token_auth_scheme = auth.OptionalHTTPBearer()


app = FastAPI(root_path=os.getenv("API_ROOT_PATH", ""))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    except (
        exceptions.ModelDoesNotExistError,
        exceptions.ImageDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post(
    "/groundtruth-segmentations", dependencies=[Depends(token_auth_scheme)]
)
def create_groundtruth_segmentations(
    data: schemas.GroundTruthSegmentationsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        logger.debug(
            f"got: {len(data.segmentations)} segmentations for dataset {data.dataset_name}"
        )
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
    data: schemas.GroundTruthClassificationsCreate,
    db: Session = Depends(get_db),
) -> list[int]:
    try:
        return crud.create_ground_truth_classifications(db=db, data=data)
    except exceptions.DatasetIsFinalizedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post(
    "/predicted-classifications", dependencies=[Depends(token_auth_scheme)]
)
def create_predicted_classifications(
    data: schemas.PredictedClassificationsCreate,
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
        return schemas.Dataset(
            **{k: getattr(dset, k) for k in schemas.Dataset.__fields__}
        )
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
        return crud.get_labels_from_dataset(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/labels/distribution",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_label_distribution_from_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.LabelDistribution]:
    try:
        return crud.get_label_distribution_from_dataset(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/info",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_info(
    dataset_name: str, db: Session = Depends(get_db)
) -> schemas.Info:
    try:
        return crud.get_dataset_info(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/images",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_images(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Image]:
    try:
        return [
            schemas.Image(
                uid=image.uid, height=image.height, width=image.width
            )
            for image in crud.get_datums_in_dataset(db, dataset_name)
        ]
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/images/{image_uid}/detections",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_image_detections(
    dataset_name: str, image_uid: str, db: Session = Depends(get_db)
) -> list[schemas.GroundTruthDetection]:
    try:
        return crud.get_groundtruth_detections_in_image(
            db, uid=image_uid, dataset_name=dataset_name
        )
    except (
        exceptions.ImageDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/images/{image_uid}/instance-segmentations",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_instance_segmentations(
    dataset_name: str, image_uid: str, db: Session = Depends(get_db)
) -> list[schemas.GroundTruthSegmentation]:
    try:
        return crud.get_groundtruth_segmentations_in_image(
            db, uid=image_uid, dataset_name=dataset_name, are_instance=True
        )
    except (
        exceptions.ImageDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/images/{image_uid}/semantic-segmentations",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_semantic_segmentations(
    dataset_name: str, image_uid: str, db: Session = Depends(get_db)
) -> list[schemas.GroundTruthSegmentation]:
    try:
        return crud.get_groundtruth_segmentations_in_image(
            db, uid=image_uid, dataset_name=dataset_name, are_instance=False
        )
    except (
        exceptions.ImageDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete(
    "/datasets/{dataset_name}", dependencies=[Depends(token_auth_scheme)]
)
def delete_dataset(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> str:
    logger.debug(f"request to delete dataset {dataset_name}")
    try:
        # make sure dataset exists
        crud.get_dataset(db, dataset_name)

        job, wrapped_fn = jobs.wrap_method_for_job(crud.delete_dataset)
        background_tasks.add_task(wrapped_fn, db=db, dataset_name=dataset_name)

        return job.uid
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
def get_model(model_name: str, db: Session = Depends(get_db)) -> schemas.Model:
    try:
        model = crud.get_model(db=db, model_name=model_name)
        return schemas.Model(
            **{k: getattr(model, k) for k in schemas.Model.__fields__}
        )
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/models/{model_name}", dependencies=[Depends(token_auth_scheme)])
def delete_model(model_name: str, db: Session = Depends(get_db)) -> None:
    return crud.delete_model(db, model_name)


@app.get(
    "/models/{model_name}/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_labels(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    try:
        return crud.get_labels_from_model(db, model_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/labels/distribution",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_label_distribution_from_model(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.ScoredLabelDistribution]:
    try:
        return crud.get_label_distribution_from_model(
            db, model_name=model_name
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/info",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_info(
    model_name: str, db: Session = Depends(get_db)
) -> schemas.Info:
    try:
        return crud.get_model_info(db, model_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/evaluation-settings",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_model_evaluations(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.EvaluationSettings]:
    try:
        return crud.get_model_evaluation_settings(db, model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluation-settings/{evaluation_settings_id}",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_evaluation_settings(
    evaluation_settings_id: str, db: Session = Depends(get_db)
):
    try:
        return crud.get_evaluation_settings_from_id(db, evaluation_settings_id)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/evaluation-settings/{evaluation_settings_id}/metrics",
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_evaluation_metrics(
    model_name: str, evaluation_settings_id: str, db: Session = Depends(get_db)
) -> list[schemas.Metric]:
    # TODO: verify evaluation_settings_id corresponds to given model_name
    return crud.get_metrics_from_evaluation_settings_id(
        db, evaluation_settings_id
    )


@app.get(
    "/models/{model_name}/evaluation-settings/{evaluation_settings_id}/confusion-matrices",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_model_confusion_matrices(
    evaluation_settings_id: str, db: Session = Depends(get_db)
) -> list[schemas.ConfusionMatrixResponse]:
    return crud.get_confusion_matrices_from_evaluation_settings_id(
        db, evaluation_settings_id
    )


@app.post(
    "/ap-metrics", status_code=202, dependencies=[Depends(token_auth_scheme)]
)
def create_ap_metrics(
    data: schemas.APRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateAPMetricsResponse:
    try:
        (
            missing_pred_labels,
            ignored_pred_labels,
        ) = crud.validate_create_ap_metrics(db, request_info=data)

        job, wrapped_fn = jobs.wrap_metric_computation(crud.create_ap_metrics)
        cm_resp = schemas.CreateAPMetricsResponse(
            missing_pred_labels=missing_pred_labels,
            ignored_pred_labels=ignored_pred_labels,
            job_id=job.uid,
        )

        background_tasks.add_task(
            wrapped_fn,
            db=db,
            request_info=data,
        )

        return cm_resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetIsDraftError,
        exceptions.InferencesAreNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))


@app.post(
    "/clf-metrics", status_code=202, dependencies=[Depends(token_auth_scheme)]
)
def create_clf_metrics(
    data: schemas.ClfOrSemanticSegMetricsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateClfOrSemanticSegMetricsResponse:
    try:
        (
            missing_pred_keys,
            ignored_pred_keys,
        ) = crud.validate_create_clf_metrics(db, request_info=data)

        job, wrapped_fn = jobs.wrap_metric_computation(crud.create_clf_metrics)

        cm_resp = schemas.CreateClfOrSemanticSegMetricsResponse(
            missing_pred_keys=missing_pred_keys,
            ignored_pred_keys=ignored_pred_keys,
            job_id=job.uid,
        )

        background_tasks.add_task(wrapped_fn, db=db, request_info=data)

        return cm_resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetIsDraftError,
        exceptions.InferencesAreNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))


@app.get("/labels", status_code=200, dependencies=[Depends(token_auth_scheme)])
def get_labels(db: Session = Depends(get_db)) -> list[schemas.Label]:
    return crud.get_all_labels(db)


@app.get("/user")
def user(
    token: HTTPAuthorizationCredentials | None = Depends(token_auth_scheme),
) -> schemas.User:
    token_payload = auth.verify_token(token)
    return schemas.User(email=token_payload.get("email"))


@app.get("/jobs/{job_id}", dependencies=[Depends(token_auth_scheme)])
def get_job(job_id: str) -> schemas.Job:
    try:
        return jobs.get_job(job_id)
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/jobs/{job_id}/metrics",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_metrics(
    job_id: str, db: Session = Depends(get_db)
) -> list[schemas.Metric]:
    try:
        job = jobs.get_job(job_id)
        if job.status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics for job since its status is {job.status}",
            )
        return crud.get_metrics_from_evaluation_settings_id(
            db, job.evaluation_settings_id
        )
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/jobs/{job_id}/confusion-matrices",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_confusion_matrices(
    job_id: str, db: Session = Depends(get_db)
) -> list[schemas.ConfusionMatrixResponse]:
    try:
        job = jobs.get_job(job_id)
        if job.status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics for job since its status is {job.status}",
            )
        return crud.get_confusion_matrices_from_evaluation_settings_id(
            db, job.evaluation_settings_id
        )

    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/jobs/{job_id}/settings",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_settings(
    job_id: str, db: Session = Depends(get_db)
) -> schemas.EvaluationSettings:
    try:
        job = jobs.get_job(job_id)
        if job.status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No settings for job since its status is {job.status}",
            )
        return crud.get_evaluation_settings_from_id(
            db, job.evaluation_settings_id
        )
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/models/{model_name}/inferences/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def finalize_inferences(
    model_name: str, dataset_name: str, db: Session = Depends(get_db)
):
    try:
        crud.finalize_inferences(
            db, model_name=model_name, dataset_name=dataset_name
        )
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))

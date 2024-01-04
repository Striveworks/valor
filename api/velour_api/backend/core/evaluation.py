from sqlalchemy import and_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from velour_api import enums, exceptions, schemas
from velour_api.backend import models, core


def _validate_evaluation_job(job_request: schemas.EvaluationJob):
    if (
        job_request.settings.filters.dataset_names is not None
        or job_request.settings.filters.dataset_metadata is not None
        or job_request.settings.filters.dataset_geospatial is not None
        or job_request.settings.filters.models_names is not None
        or job_request.settings.filters.models_metadata is not None
        or job_request.settings.filters.models_geospatial is not None
        or job_request.settings.filters.prediction_scores is not None
        or job_request.settings.filters.task_types is not None
    ):
        raise ValueError(
            "Evaluation filter objects should not include any dataset, model, prediction score or task type filters."
        )
    elif (
        job_request.task_type == enums.TaskType.SEGMENTATION
        and job_request.settings.filters.annotation_types is not None
    ):
        raise ValueError(
            "Segmentation evaluation should not include any annotation type filters."
        )


def create_or_get_evaluation(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> int:
    """
    Creates an evaluation.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The evaluation job to create.

    Returns
    -------
    int
        The id of the new evaluation.
    """
    # validate args
    if (
        not isinstance(db, Session)
        or not isinstance(job_request, schemas.EvaluationJob)
    ):
        raise TypeError

    # validate parameters
    match job_request.task_type:
        case enums.TaskType.DETECTION:
            if not job_request.settings.parameters:
                job_request.settings.parameters = schemas.DetectionParameters()
        case _:
            if job_request.settings.parameters:
                raise ValueError(
                    f"Evaluations with task type `{job_request.task_type}` do not take parametric input."
                )

    # validate filters
    if not job_request.settings.filters:
        job_request.settings.filters = schemas.Filter()
    else:
        _validate_evaluation_job(job_request=job_request)

    dataset = core.fetch_dataset(db, job_request.dataset)
    model = core.fetch_model(db, job_request.model)

    evaluation = (
        db.query(models.Evaluation)
        .where(
            and_(
                models.Evaluation.dataset_id == dataset.id,
                models.Evaluation.model_id == model.id,
                models.Evaluation.task_type == job_request.task_type,
                models.Evaluation.settings == job_request.settings.model_dump(),
            )
        )
        .one_or_none()
    )

    if evaluation is None:
        try:
            evaluation = models.Evaluation(
                dataset_id=dataset.id,
                model_id=model.id,
                task_type=job_request.task_type,
                settings=job_request.settings.model_dump(),
                status=enums.JobStatus.CREATING,
            )
            db.add(evaluation)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise exceptions.EvaluationAlreadyExistsError()
    
    return evaluation.id


def set_evaluation_status(
    db: Session,
    evaluation_id: int,
    status: enums.JobStatus,
):
    """
    Sets the status of an evaluation.
    """
    pass
    # evaluation = fetch_evaluation(db, evaluation_id)
    # try:
    #     evaluation.status = status
    #     db.commit()
    # except Exception as e:
    #     db.rollback()
    #     raise e
    

def _get_annotation_types_for_computation(
    db: Session,
    dataset: models.Dataset,
    model: models.Model,
    job_filter: schemas.Filter | None = None,
) -> enums.AnnotationType:
    """Fetch the groundtruth and prediction annotation types for a given dataset / model combination."""
    # get dominant type
    groundtruth_type = core.get_annotation_type(db, dataset, None)
    prediction_type = core.get_annotation_type(db, dataset, model)
    greatest_common_type = (
        groundtruth_type
        if groundtruth_type < prediction_type
        else prediction_type
    )
    if job_filter.annotation_types:
        if greatest_common_type not in job_filter.annotation_types:
            sorted_types = sorted(
                job_filter.annotation_types,
                key=lambda x: x,
                reverse=True,
            )
            for annotation_type in sorted_types:
                if greatest_common_type >= annotation_type:
                    return annotation_type, annotation_type
            raise RuntimeError(
                f"Annotation type filter is too restrictive. Attempted filter `{greatest_common_type}` over `{groundtruth_type, prediction_type}`."
            )
    return groundtruth_type, prediction_type
    

def get_disjoint_labels_from_evaluation(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> tuple:
    """Return a tuple containing the unique labels associated with the groundtruths and predictions stored in a database."""

    # load sql objects
    dataset = core.fetch_dataset(db, job_request.dataset)
    model = core.fetch_model(db, job_request.model)

    # get filter object
    if not job_request.settings.filters:
        filters = schemas.Filter()
    else:
        _validate_evaluation_job(job_request=job_request)
        filters = job_request.settings.filters.model_copy()

    # determine annotation types
    (
        groundtruth_type,
        prediction_type,
    ) = _get_annotation_types_for_computation(
        db, dataset, model, filters
    )

    # create groundtruth label filter
    groundtruth_label_filter = filters.model_copy()
    groundtruth_label_filter.dataset_names = [job_request.dataset]
    groundtruth_label_filter.annotation_types = [groundtruth_type]

    # create prediction label filter
    prediction_label_filter = filters.model_copy()
    prediction_label_filter.dataset_names = [job_request.dataset]
    prediction_label_filter.models_names = [model.name]
    prediction_label_filter.annotation_types = [prediction_type]

    # TODO - update core.get_disjoint_labels to take filter object
    groundtruth_labels = core.get_labels(
        db, groundtruth_label_filter, ignore_predictions=True
    )
    prediction_labels = core.get_labels(
        db, prediction_label_filter, ignore_groundtruths=True
    )
    groundtruth_unique = list(groundtruth_labels - prediction_labels)
    prediction_unique = list(prediction_labels - groundtruth_labels)

    return groundtruth_unique, prediction_unique

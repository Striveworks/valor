from sqlalchemy.orm import Session

from velour_api import enums, backend, schemas
from velour_api.backend import set_dataset_status, set_model_status


def finalize(*, db: Session, dataset_name: str, model_name: str = None):
    """
    Finalizes dataset and dataset/model pairings.
    """
    if dataset_name and model_name:
        set_model_status(
            db=db,
            dataset_name=dataset_name,
            model_name=model_name,
            status=enums.TableStatus.FINALIZED,
        )
    elif dataset_name:
        set_dataset_status(
            db=db,
            name=dataset_name,
            status=enums.TableStatus.FINALIZED,
        )


def compute_clf_metrics(
    *,
    db: Session,
    evaluation_id: int,
    job_request: schemas.EvaluationJob,
):
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id: int
        The job id.
    job_request: schemas.EvaluationJob
        The evaluation job.
    """
    backend.compute_clf_metrics(
        db=db,
        evaluation_id=evaluation_id,
    )


def compute_detection_metrics(
    *,
    db: Session,
    evaluation_id: int,
    job_request: schemas.EvaluationJob,
):
    """
    Compute detection metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationJob
        The evaluation job object.
    evaluation_id: int
        The job id.
    """
    backend.compute_detection_metrics(
        db=db,
        evaluation_id=evaluation_id,
    )



def compute_semantic_segmentation_metrics(
    *,
    db: Session,
    evaluation_id: int,
    job_request: schemas.EvaluationJob,
):
    """
    Compute semantic segmentation metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationJob
        The evaluation job object.
    evaluation_id: int
        The job id.
    """
    backend.compute_semantic_segmentation_metrics(
        db=db,
        evaluation_id=evaluation_id,
        job_request=job_request,
    )
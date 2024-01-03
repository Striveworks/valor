import numpy as np
from sqlalchemy import Float, Integer
from sqlalchemy.orm import Bundle, Session
from sqlalchemy.sql import and_, func, select

from velour_api import schemas
from velour_api.backend import core, models
from velour_api.backend.metrics.metric_utils import (
    create_metric_mappings,
    get_or_create_row,
)
from velour_api.backend.ops import Query
from velour_api.enums import TaskType


def _compute_binary_roc_auc(
    db: Session,
    job_request: schemas.EvaluationJob,
    label: schemas.Label,
) -> float:
    """
    Computes the binary ROC AUC score of a dataset and label.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The job request to compute the score for.
    label : schemas.Label
        The label to compute the score for.

    Returns
    -------
    float
        The binary ROC AUC score.
    """

    # query to get the datum_ids and label values of groundtruths that have the given label key
    gts_filter = job_request.settings.filters.model_copy()
    gts_filter.dataset_names = [job_request.dataset]
    gts_filter.label_keys = [label.key]

    gts_query = (
        Query(
            models.Annotation.datum_id.label("datum_id"),
            models.Label.value.label("label_value"),
        )
        .filter(gts_filter)
        .groundtruths("groundtruth_subquery")
    )

    # get the prediction scores for the given label (key and value)
    preds_filter = job_request.settings.filters.model_copy()
    preds_filter.dataset_names = [job_request.dataset]
    preds_filter.models_names = [job_request.model]
    preds_filter.labels = [{label.key: label.value}]

    preds_query = (
        Query(
            models.Annotation.datum_id.label("datum_id"),
            models.Prediction.score.label("score"),
            models.Label.value.label("label_value"),
        )
        .filter(preds_filter)
        .predictions("prediction_subquery")
    )

    # number of groundtruth labels that match the given label value
    n_pos = db.scalar(
        select(func.count(gts_query.c.label_value)).where(
            gts_query.c.label_value == label.value
        )
    )
    # total number of groundtruths
    n = db.scalar(select(func.count(gts_query.c.label_value)))

    if n - n_pos == 0:
        return 1.0

    # true positive rates
    tprs = (
        func.sum(
            (gts_query.c.label_value == label.value).cast(Integer).cast(Float)
        ).over(order_by=-preds_query.c.score)
        / n_pos
    )

    # false positive rates
    fprs = func.sum(
        (gts_query.c.label_value != label.value).cast(Integer).cast(Float)
    ).over(order_by=-preds_query.c.score) / (n - n_pos)

    tprs_fprs_query = (
        select(
            tprs.label("tprs"),
            fprs.label("fprs"),
            preds_query.c.score,
        ).join(preds_query, gts_query.c.datum_id == preds_query.c.datum_id)
        # .order_by(preds_query.c.score.desc())
    ).subquery()

    trap_areas = select(
        (
            0.5
            * (
                tprs_fprs_query.c.tprs
                + func.lag(tprs_fprs_query.c.tprs).over(
                    order_by=-tprs_fprs_query.c.score
                )
            )
            * (
                tprs_fprs_query.c.fprs
                - func.lag(tprs_fprs_query.c.fprs).over(
                    order_by=-tprs_fprs_query.c.score
                )
            )
        ).label("trap_area")
    ).subquery()

    ret = db.scalar(func.sum(trap_areas.c.trap_area))
    if ret is None:
        return np.nan
    return ret


def _compute_roc_auc(
    db: Session,
    job_request: schemas.EvaluationJob,
    label_key: str,
) -> float:
    """
    Computes the area under the ROC curve. Note that for the multi-class setting
    this does one-vs-rest AUC for each class and then averages those scores. This should give
    the same thing as `sklearn.metrics.roc_auc_score` with `multi_class="ovr"`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The job request to compute the score for.
    label_key : str
        The label key to compute the score for.

    Returns
    -------
    float
        The ROC AUC.
    """

    label_filter = job_request.settings.filters.model_copy()
    label_filter.dataset_names = [job_request.dataset]
    label_filter.label_keys = [label_key]

    labels = {
        schemas.Label(key=label.key, value=label.value)
        for label in db.query(
            Query(models.Label).filter(label_filter).groundtruths()
        ).all()
    }
    if len(labels) == 0:
        raise RuntimeError(
            f"The label key '{label_key}' is not a classification label in the dataset {job_request.dataset}."
        )

    sum_roc_aucs = 0
    label_count = 0
    for label in labels:
        bin_roc = _compute_binary_roc_auc(db, job_request, label)

        if bin_roc is not None:
            sum_roc_aucs += bin_roc
            label_count += 1

    return sum_roc_aucs / label_count


def _compute_confusion_matrix_at_label_key(
    db: Session,
    job_request: schemas.EvaluationJob,
    label_key: str,
) -> schemas.ConfusionMatrix | None:
    """
    Computes the confusion matrix at a label_key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The job request to compute the matrix for.
    label_key : str
        The label key to compute the matrix for.

    Returns
    -------
    schemas.ConfusionMatrix | None
        Returns None in the case that there are no common images in the dataset
        that have both a groundtruth and prediction with label key `label_key`. Otherwise
        returns the confusion matrix.
    """
    if job_request.settings.filters is None:
        job_request.settings.filters = schemas.Filter()

    # groundtruths filter
    gFilter = job_request.settings.filters.model_copy()
    gFilter.dataset_names = [job_request.dataset]
    gFilter.label_keys = [label_key]

    # predictions filter
    pFilter = job_request.settings.filters.model_copy()
    pFilter.dataset_names = [job_request.dataset]
    pFilter.models_names = [job_request.model]
    pFilter.label_keys = [label_key]

    # 0. Get groundtruths that conform to gFilter
    groundtruths = (
        Query(
            models.GroundTruth,
            models.Annotation.datum_id.label("datum_id"),
        )
        .filter(gFilter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    # 1. Get predictions that conform to pFilter
    predictions = (
        Query(models.Prediction)
        .filter(pFilter)
        .predictions(as_subquery=False)
        .alias()
    )

    # 2. Get the max prediction scores by datum
    max_scores_by_datum_id = (
        select(
            func.max(predictions.c.score).label("max_score"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .join(
            models.Annotation,
            models.Annotation.id == predictions.c.annotation_id,
        )
        .group_by(models.Annotation.datum_id)
        .alias()
    )

    # 3. Remove duplicate scores per datum
    # used for the edge case where the max confidence appears twice
    # the result of this query is all of the hard predictions
    min_id_query = (
        select(
            func.min(predictions.c.id).label("min_id"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .select_from(predictions)
        .join(
            models.Annotation,
            models.Annotation.id == predictions.c.annotation_id,
        )
        .join(
            max_scores_by_datum_id,
            and_(
                models.Annotation.datum_id
                == max_scores_by_datum_id.c.datum_id,
                predictions.c.score == max_scores_by_datum_id.c.max_score,
            ),
        )
        .group_by(models.Annotation.datum_id)
        .alias()
    )

    # 4. Get labels for hard predictions, organize per datum
    hard_preds_query = (
        select(
            models.Label.value.label("pred_label_value"),
            min_id_query.c.datum_id.label("datum_id"),
        )
        .select_from(min_id_query)
        .join(
            models.Prediction,
            models.Prediction.id == min_id_query.c.min_id,
        )
        .join(
            models.Label,
            models.Label.id == models.Prediction.label_id,
        )
        .alias()
    )

    # 5. Link value to the Label.value object
    b = Bundle("cols", hard_preds_query.c.pred_label_value, models.Label.value)

    # 6. Generate confusion matrix
    total_query = (
        select(b, func.count())
        .select_from(hard_preds_query)
        .join(
            groundtruths,
            groundtruths.c.datum_id == hard_preds_query.c.datum_id,
        )
        .join(
            models.Label,
            models.Label.id == groundtruths.c.label_id,
        )
        .group_by(b)
    )

    res = db.execute(total_query).all()

    if len(res) == 0:
        # this means there's no predictions and groundtruths with the label key
        # for the same image
        return None

    return schemas.ConfusionMatrix(
        label_key=label_key,
        entries=[
            schemas.ConfusionMatrixEntry(
                prediction=r[0][0], groundtruth=r[0][1], count=r[1]
            )
            for r in res
        ],
    )


def _compute_accuracy_from_cm(cm: schemas.ConfusionMatrix) -> float:
    """
    Computes the accuracy score from a confusion matrix.

    Parameters
    ----------
    cm : schemas.ConfusionMatrix
        The confusion matrix to use.

    Returns
    ----------
    float
        The resultant accuracy score.
    """
    return cm.matrix.trace() / cm.matrix.sum()


def _compute_precision_and_recall_f1_from_confusion_matrix(
    cm: schemas.ConfusionMatrix, label_value: str
) -> tuple[float, float, float]:
    """
    Computes the precision, recall, and f1 score at a class index

    Parameters
    ----------
    cm : schemas.ConfusionMatrix
        The confusion matrix to use.
    label_key : str
        The label key to compute scores for.

    Returns
    ----------
    Tuple[float, float, float]
        A tuple containing the precision, recall, and F1 score.
    """
    cm_matrix = cm.matrix
    if label_value not in cm.label_map:
        return np.nan, np.nan, np.nan
    class_index = cm.label_map[label_value]

    true_positives = cm_matrix[class_index, class_index]
    # number of times the class was predicted
    n_preds = cm_matrix[:, class_index].sum()
    n_gts = cm_matrix[class_index, :].sum()

    prec = true_positives / n_preds if n_preds else 0
    recall = true_positives / n_gts if n_gts else 0

    f1_denom = prec + recall
    if f1_denom == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / f1_denom
    return prec, recall, f1


def _compute_confusion_matrix_and_metrics_at_label_key(
    db: Session,
    job_request: schemas.EvaluationJob,
    label_key: str,
    labels: list[models.Label],
) -> (
    tuple[
        schemas.ConfusionMatrix,
        list[
            schemas.AccuracyMetric
            | schemas.ROCAUCMetric
            | schemas.PrecisionMetric
            | schemas.RecallMetric
            | schemas.F1Metric
        ],
    ]
    | None
):
    """
    Computes the confusion matrix and all metrics for a given label key.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The job request to compute the matrix for.
    label_key : str
        The label key to compute the matrix for.
    labels
        All of the labels in both the groundtruths and predictions that have key
        equal to `label_key`.

    Returns
    -------
    tuple[schemas.ConfusionMatrix, list[schemas.AccuracyMetric | schemas.ROCAUCMetric | schemas.PrecisionMetric
                                        | schemas.RecallMetric | schemas.F1Metric]] | None
        Returns None if there are no predictions and groundtruths with the given label
        key for the same datum. Otherwise returns a tuple, with the first element the confusion
        matrix and the second a list of all metrics (accuracy, ROC AUC, precisions, recalls, and f1s).
    """

    for label in labels:
        if label.key != label_key:
            raise ValueError(
                f"Expected all elements of `labels` to have label key equal to {label_key} but got label {label}."
            )

    confusion_matrix = _compute_confusion_matrix_at_label_key(
        db=db,
        job_request=job_request,
        label_key=label_key,
    )

    if confusion_matrix is None:
        return None

    # aggregate metrics (over all label values)
    metrics = [
        schemas.AccuracyMetric(
            label_key=label_key,
            value=_compute_accuracy_from_cm(confusion_matrix),
        ),
        schemas.ROCAUCMetric(
            label_key=label_key,
            value=_compute_roc_auc(
                db,
                job_request,
                label_key,
            ),
        ),
    ]

    # metrics that are per label
    for label in labels:
        (
            precision,
            recall,
            f1,
        ) = _compute_precision_and_recall_f1_from_confusion_matrix(
            confusion_matrix, label.value
        )

        pydantic_label = schemas.Label(key=label.key, value=label.value)
        metrics.append(
            schemas.PrecisionMetric(
                label=pydantic_label,
                value=precision,
            )
        )
        metrics.append(
            schemas.RecallMetric(
                label=pydantic_label,
                value=recall,
            )
        )
        metrics.append(
            schemas.F1Metric(
                label=pydantic_label,
                value=f1,
            )
        )

    return confusion_matrix, metrics


def _compute_clf_metrics(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> tuple[
    list[schemas.ConfusionMatrix],
    list[
        schemas.ConfusionMatrix
        | schemas.AccuracyMetric
        | schemas.ROCAUCMetric
        | schemas.PrecisionMetric
        | schemas.RecallMetric
        | schemas.F1Metric
    ],
]:
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The job request to compute metrics for.

    Returns
    ----------
    Tuple[List[schemas.ConfusionMatrix], List[schemas.ConfusionMatrix | schemas.AccuracyMetric | schemas.ROCAUCMetric| schemas.PrecisionMetric | schemas.RecallMetric | schemas.F1Metric]]
        A tuple of confusion matrices and metrics.
    """
    # construct dataset filter
    groundtruth_label_filter = job_request.settings.filters.model_copy()
    groundtruth_label_filter.dataset_names = [job_request.dataset]

    # construct model filter
    prediction_label_filter = job_request.settings.filters.model_copy()
    prediction_label_filter.dataset_names = [job_request.dataset]
    prediction_label_filter.models_names = [job_request.model]

    # retrieve dataset labels
    dataset_labels = {
        schemas.Label(key=label.key, value=label.value)
        for label in db.query(
            Query(models.Label).filter(groundtruth_label_filter).groundtruths()
        ).all()
    }

    # retrieve model labels
    model_labels = {
        schemas.Label(key=label.key, value=label.value)
        for label in db.query(
            Query(models.Label).filter(prediction_label_filter).predictions()
        ).all()
    }

    # get union of labels + unique keys
    labels = list(dataset_labels.union(model_labels))
    unique_label_keys = set([label.key for label in labels])

    # compute metrics and confusion matrix for each label key
    confusion_matrices, metrics = [], []
    for label_key in unique_label_keys:
        cm_and_metrics = _compute_confusion_matrix_and_metrics_at_label_key(
            db,
            job_request=job_request,
            label_key=label_key,
            labels=[label for label in labels if label.key == label_key],
        )
        if cm_and_metrics is not None:
            confusion_matrices.append(cm_and_metrics[0])
            metrics.extend(cm_and_metrics[1])

    return confusion_matrices, metrics


def create_clf_evaluation(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> int:
    """
    Create a classification evaluation job.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The job request to create an evaluation for.

    Returns
    ----------
    int
        The evaluation job id.

    Raises
    ----------
    TypeError
        If the job's task type is incorrect.
    ValueError
        If the evaluation contains an inappropriate filter.
    """
    # check matching task_type
    if job_request.task_type != TaskType.CLASSIFICATION:
        raise TypeError(
            "Invalid task_type, please choose an evaluation method that supports classification"
        )

    # configure filters object
    if job_request.settings.filters:
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

    # create evaluation row
    dataset = core.fetch_dataset(db, job_request.dataset)
    model = core.fetch_model(db, job_request.model)
    es = get_or_create_row(
        db,
        models.Evaluation,
        mapping={
            "dataset_id": dataset.id,
            "model_id": model.id,
            "task_type": TaskType.CLASSIFICATION,
            "settings": job_request.settings.model_dump(),
        },
    )
    return es.id


def create_clf_metrics(
    db: Session,
    job_id: int,
) -> int:
    """
    Create classification metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_id : int
        The job ID to create metrics for.

    Returns
    ----------
    int
        The evaluation job id.
    """
    evaluation = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == job_id)
    )

    # unpack job request
    job_request = schemas.EvaluationJob(
        dataset=evaluation.dataset.name,
        model=evaluation.model.name,
        task_type=evaluation.task_type,
        settings=schemas.EvaluationSettings(**evaluation.settings),
        id=evaluation.id,
    )

    # configure filters
    if not job_request.settings.filters:
        job_request.settings.filters = schemas.Filter()
    job_request.settings.filters.task_types = [TaskType.CLASSIFICATION]

    confusion_matrices, metrics = _compute_clf_metrics(
        db=db,
        job_request=job_request,
    )

    confusion_matrices_mappings = create_metric_mappings(
        db=db,
        metrics=confusion_matrices,
        evaluation_id=job_request.id,
    )

    for mapping in confusion_matrices_mappings:
        get_or_create_row(
            db,
            models.ConfusionMatrix,
            mapping,
        )

    metric_mappings = create_metric_mappings(
        db=db, metrics=metrics, evaluation_id=job_request.id
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empirically noticed value can slightly change due to floating
        # point errors
        get_or_create_row(
            db,
            models.Metric,
            mapping,
            columns_to_ignore=["value"],
        )

    return job_id

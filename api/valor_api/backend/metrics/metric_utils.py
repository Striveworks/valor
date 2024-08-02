import json
from collections import defaultdict
from typing import Any, Callable, Sequence

from sqlalchemy import ColumnElement, Label, and_, case, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from valor_api import enums, logger, schemas
from valor_api.backend import core, models
from valor_api.backend.query import generate_select
from valor_api.exceptions import InvalidLLMResponseError

LabelMapType = list[list[list[str]]]


def profiler(fn: Callable):
    def wrapper(*args, **kwargs):
        import time

        print(f">>>> {fn.__name__}")
        start = time.time()
        result = fn(*args, **kwargs)
        print(f"<<<< {fn.__name__} - {round(time.time() - start, 2)}")
        return result

    return wrapper


@profiler
def create_label_mapping(
    db: Session,
    labels: list[models.Label],
    label_map: LabelMapType | None,
) -> ColumnElement[bool] | Label[int]:
    """
    Creates a dictionary of mappings that connect each label with a "grouper" (i.e., a unique ID-key-value combination that can represent one or more labels).
    These mappings enable Valor to group multiple labels together using the label_map argument in each evaluation function.

    Parameters
    ----------
    db : Session
        The database session.
    labels : list[models.Label]
        A list of labels that exist for this evaluation job.
    label_map: LabelMapType, optional
        An optional label map to use when grouping labels. If None is passed, this function will still create the appropriate mappings using individual labels.

    Returns
    ----------
    ColumnElement[bool] | Label[int]
        A label id statement.
    """

    if label_map:
        # add grouper labels to database (if they don't exist)
        existing_labels = {(label.key, label.value) for label in labels}
        mapping_dict = {
            tuple(label): tuple(grouper) for label, grouper in label_map
        }
        grouper_labels = set(mapping_dict.values())
        missing_grouper_labels = grouper_labels - existing_labels
        core.create_labels(
            db=db,
            labels=[
                schemas.Label(key=key, value=value)
                for key, value in missing_grouper_labels
            ],
        )

        # cache label ids
        all_labels = grouper_labels.union(existing_labels)
        map_label_to_id = {
            (label.key, label.value): label.id
            for label in db.query(models.Label)
            .where(
                or_(
                    *[
                        and_(
                            models.Label.key == label[0],
                            models.Label.value == label[1],
                        )
                        for label in all_labels
                    ]
                )
            )
            .all()
        }

        # create label id mapping
        label_mapping = [
            (
                models.Label.id == map_label_to_id[label],  # type: ignore - pyright doesnt see tuple[str, str]
                map_label_to_id[grouper],  # type: ignore - pyright doesnt see tuple[str, str]
            )
            for label, grouper in mapping_dict.items()
        ]

        return case(
            *label_mapping,
            else_=models.Label.id,
        ).label("label_id")
    else:
        return models.Label.id.label("label_id")


@profiler
def commit_results(
    db: Session,
    metrics: Sequence[
        schemas.APMetric
        | schemas.ARMetric
        | schemas.APMetricAveragedOverIOUs
        | schemas.mAPMetric
        | schemas.mARMetric
        | schemas.mAPMetricAveragedOverIOUs
        | schemas.ConfusionMatrix
        | schemas.AccuracyMetric
        | schemas.ROCAUCMetric
        | schemas.PrecisionMetric
        | schemas.RecallMetric
        | schemas.F1Metric
        | schemas.IOUMetric
        | schemas.mIOUMetric
        | schemas.PrecisionRecallCurve
        | schemas.DetailedPrecisionRecallCurve
        | schemas.AnswerRelevanceMetric
        | schemas.BiasMetric
        | schemas.BLEUMetric
        | schemas.CoherenceMetric
        | schemas.ContextRelevanceMetric
        | schemas.FaithfulnessMetric
        | schemas.HallucinationMetric
        | schemas.ROUGEMetric
        | schemas.ToxicityMetric
    ],
    evaluation_id: int,
):
    """
    Create metric mappings from a list of metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    metrics : List
        A list of metrics to create mappings for.
    evaluation_id : int
        The id of the evaluation job.
    """

    # cache labels for metrics that use them
    cached_labels = defaultdict(list)
    for metric in metrics:
        if isinstance(
            metric,
            (
                schemas.APMetric,
                schemas.ARMetric,
                schemas.APMetricAveragedOverIOUs,
                schemas.PrecisionMetric,
                schemas.RecallMetric,
                schemas.F1Metric,
                schemas.IOUMetric,
            ),
        ):
            cached_labels[metric.label.key].append(metric.label.value)
    cached_label_to_id = {
        schemas.Label(key=row.key, value=row.value): row.id
        for row in (
            db.query(models.Label)
            .where(
                or_(
                    *[
                        and_(
                            models.Label.key == key,
                            models.Label.value.in_(values),
                        )
                        for key, values in cached_labels.items()
                    ]
                )
            )
            .all()
        )
    }

    metric_rows = []
    confusion_rows = []
    for metric in metrics:
        if isinstance(
            metric,
            (
                schemas.APMetric,
                schemas.ARMetric,
                schemas.APMetricAveragedOverIOUs,
                schemas.PrecisionMetric,
                schemas.RecallMetric,
                schemas.F1Metric,
                schemas.IOUMetric,
            ),
        ):
            metric_rows.append(
                models.Metric(
                    **metric.db_mapping(
                        label_id=cached_label_to_id[metric.label],
                        evaluation_id=evaluation_id,
                    )
                )
            )
        elif isinstance(metric, schemas.ConfusionMatrix):
            confusion_rows.append(
                models.ConfusionMatrix(
                    **metric.db_mapping(evaluation_id=evaluation_id)
                )
            )
        else:
            metric_rows.append(
                models.Metric(**metric.db_mapping(evaluation_id=evaluation_id))
            )

    try:
        if metric_rows:
            db.add_all(metric_rows)
        if confusion_rows:
            db.add_all(confusion_rows)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


@profiler
def log_evaluation_duration(
    db: Session,
    evaluation: models.Evaluation,
):
    """
    Store analytics regarding the evaluation's runtime in the metadata field of the evaluation table.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation : models.Evaluation
        The evaluation to log to.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    """

    server_time = db.execute(func.now()).scalar().replace(tzinfo=None)  # type: ignore - guaranteed to return server time if psql is running
    duration = (server_time - evaluation.created_at).total_seconds()

    try:
        metadata = dict(evaluation.meta) if evaluation.meta else {}
        metadata.update({"duration": duration})
        evaluation.meta = metadata
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


@profiler
def log_evaluation_item_counts(
    db: Session,
    evaluation: models.Evaluation,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
):
    """
    Store analytics regarding the number of elements processed by the evaluation in the metadata field of the evaluation table.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation : models.Evaluation
        The evaluation to log to.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    """
    # get ground truth, prediction, annotation, and label counts
    gt_subquery = generate_select(
        models.Datum.id.label("datum_id"),
        models.GroundTruth,
        filters=groundtruth_filter,
        label_source=models.GroundTruth,
    ).alias()

    gts = db.execute(
        select(
            gt_subquery.c.datum_id,
            gt_subquery.c.annotation_id,
            gt_subquery.c.label_id,
        ).select_from(gt_subquery)
    ).all()

    # handle edge case where no gts come back
    if not gts:
        gt_datums, gt_annotation_id, gt_label_id = set(), set(), set()
    else:
        gt_datums, gt_annotation_id, gt_label_id = map(set, zip(*gts))

    pd_subquery = generate_select(
        models.Datum.id.label("datum_id"),
        models.Prediction,
        filters=prediction_filter,
        label_source=models.Prediction,
    ).alias()

    pds = db.execute(
        select(
            pd_subquery.c.datum_id,
            pd_subquery.c.annotation_id,
            pd_subquery.c.label_id,
        ).select_from(pd_subquery)
    ).all()

    if not pds:
        pd_datums, pd_annotation_id, pd_label_id = set(), set(), set()
    else:
        pd_datums, pd_annotation_id, pd_label_id = map(set, zip(*pds))

    datum_cnt = len(gt_datums | pd_datums)
    annotation_cnt = len(gt_annotation_id | pd_annotation_id)
    label_cnt = len(gt_label_id | pd_label_id)

    output = {
        "annotations": annotation_cnt,
        "labels": label_cnt,
        "datums": datum_cnt,
    }

    try:
        metadata = dict(evaluation.meta) if evaluation.meta else {}
        metadata.update(output)
        evaluation.meta = metadata
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def validate_computation(fn: Callable) -> Callable:
    """
    Computation decorator that validates that a computation can proceed.
    """

    def wrapper(*args, **kwargs):
        if "db" not in kwargs:
            raise RuntimeError(
                "This decorator requires `db` to be explicitly defined in kwargs."
            )
        if "evaluation_id" not in kwargs:
            raise RuntimeError(
                "This decorator requires `evaluation_id` to be explicitly defined in kwargs."
            )

        db = kwargs["db"]
        evaluation_id = kwargs["evaluation_id"]

        if not isinstance(db, Session):
            raise TypeError(
                "Expected `db` to be of type `sqlalchemy.orm.Session`."
            )
        if not isinstance(evaluation_id, int):
            raise TypeError("Expected `evaluation_id` to be of type `int`.")

        # edge case - evaluation has already been run
        if core.get_evaluation_status(db, evaluation_id) not in [
            enums.EvaluationStatus.PENDING,
            enums.EvaluationStatus.FAILED,
        ]:
            return evaluation_id

        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.RUNNING
        )
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            core.set_evaluation_status(
                db, evaluation_id, enums.EvaluationStatus.FAILED
            )
            logger.error(
                f"Valor Exception: Evaluation '{evaluation_id}'",
                method=fn.__name__,
                exc_info=e,
            )
            raise e
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.DONE
        )
        return result

    return wrapper


@profiler
def prepare_filter_for_evaluation(
    filters: schemas.Filter,
    dataset_names: list[str],
    model_name: str,
    task_type: enums.TaskType,
) -> tuple[schemas.Filter, schemas.Filter]:
    """
    Prepares the filter for use by an evaluation method.

    This function will be expanded in a future PR.

    Parameters
    ----------
    filters : Filter
        The data filter.
    dataset_names : list[str]
        A list of dataset names to filter by.
    model_name : str
        A model name to filter by.
    task_type : TaskType
        A task type to filter by.

    Returns
    -------
    Filter
        A filter ready for evaluation.
    """

    # create dataset constraint
    dataset_conditions = schemas.LogicalFunction.or_(
        *[
            schemas.Condition(
                lhs=schemas.Symbol(name=schemas.SupportedSymbol.DATASET_NAME),
                rhs=schemas.Value.infer(name),
                op=schemas.FilterOperator.EQ,
            )
            for name in dataset_names
        ]
    )

    # create model constraint
    model_condition = schemas.Condition(
        lhs=schemas.Symbol(name=schemas.SupportedSymbol.MODEL_NAME),
        rhs=schemas.Value.infer(model_name),
        op=schemas.FilterOperator.EQ,
    )

    # create task type constraint
    task_type_condition = schemas.Condition(
        lhs=schemas.Symbol(name=schemas.SupportedSymbol.TASK_TYPE),
        rhs=schemas.Value(
            type=schemas.SupportedType.TASK_TYPE, value=task_type
        ),
        op=schemas.FilterOperator.CONTAINS,
    )

    # create new annotations filter
    filters.annotations = (
        schemas.LogicalFunction.and_(
            filters.annotations,
            task_type_condition,
            dataset_conditions,
        )
        if filters.annotations
        else task_type_condition
    )

    if task_type == enums.TaskType.TEXT_GENERATION:

        filters.groundtruths = None
        filters.predictions = None

        # create new annotations filter
        groundtruth_filter = filters.model_copy()

        predictions_filter = filters.model_copy()
        predictions_filter.annotations = (
            schemas.LogicalFunction.and_(
                predictions_filter.annotations,
                model_condition,
            )
            if predictions_filter.annotations
            else model_condition
        )

    else:

        # create new groundtruth filter
        filters.groundtruths = (
            schemas.LogicalFunction.and_(
                filters.groundtruths,
                dataset_conditions,
            )
            if filters.groundtruths
            else dataset_conditions
        )

        # create new prediction filter
        filters.predictions = (
            schemas.LogicalFunction.and_(
                filters.predictions,
                dataset_conditions,
                model_condition,
            )
            if filters.predictions
            else schemas.LogicalFunction.and_(
                dataset_conditions,
                model_condition,
            )
        )

        groundtruth_filter = filters.model_copy()
        groundtruth_filter.predictions = None

        predictions_filter = filters.model_copy()
        predictions_filter.groundtruths = None

    return (groundtruth_filter, predictions_filter)


def trim_and_load_json(input_string: str) -> Any:
    """
    Trims and loads input_string as a json. Adapted from DeepEval https://github.com/confident-ai/deepeval/blob/dc117a5ea2160dbb61909c537908a41f7da4dfe7/deepeval/metrics/utils.py#L50

    Parameters
    ----------
    input_string : str
        The input string to trim and load as a json.

    Returns
    -------
    Any
        The json object.
    """
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError as e:
        raise InvalidLLMResponseError(
            "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model. JSONDecodeError: "
            + str(e)
        )

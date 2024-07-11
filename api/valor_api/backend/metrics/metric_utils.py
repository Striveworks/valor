from collections import defaultdict
from typing import Callable, Sequence

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from valor_api import enums, logger, schemas
from valor_api.backend import core, models
from valor_api.backend.query import generate_select

LabelMapType = list[list[list[str]]]


def _create_detection_grouper_mappings(
    mapping_dict: dict[tuple[str, ...], tuple[str, ...]],
    labels: list[models.Label],
) -> dict[str, dict]:
    """Create grouper mappings for use when evaluating detections."""

    label_id_to_grouper_id_mapping = {}
    label_id_to_grouper_key_mapping = {}
    grouper_id_to_grouper_label_mapping = {}
    grouper_id_to_label_ids_mapping = defaultdict(list)

    for label in labels:
        mapped_key, mapped_value = mapping_dict.get(
            (label.key, label.value), (label.key, label.value)
        )
        # create an integer to track each group by
        grouper_id = hash((mapped_key, mapped_value))
        # create a separate grouper_key_id which is used to cross-join labels that share a given key
        # when computing IOUs for PrecisionRecallCurve
        grouper_key_id = mapped_key

        label_id_to_grouper_id_mapping[label.id] = grouper_id
        label_id_to_grouper_key_mapping[label.id] = grouper_key_id
        grouper_id_to_grouper_label_mapping[grouper_id] = schemas.Label(
            key=mapped_key, value=mapped_value
        )
        grouper_id_to_label_ids_mapping[grouper_id].append(label.id)

    return {
        "label_id_to_grouper_id_mapping": label_id_to_grouper_id_mapping,
        "label_id_to_grouper_key_mapping": label_id_to_grouper_key_mapping,
        "grouper_id_to_label_ids_mapping": grouper_id_to_label_ids_mapping,
        "grouper_id_to_grouper_label_mapping": grouper_id_to_grouper_label_mapping,
    }


def _create_segmentation_grouper_mappings(
    mapping_dict: dict[tuple[str, ...], tuple[str, ...]],
    labels: list[models.Label],
) -> dict[str, dict]:
    """Create grouper mappings for use when evaluating segmentations."""

    grouper_id_to_grouper_label_mapping = {}
    grouper_id_to_label_ids_mapping = defaultdict(list)

    for label in labels:
        mapped_key, mapped_value = mapping_dict.get(
            (label.key, label.value), (label.key, label.value)
        )
        # create an integer to track each group by
        grouper_id = hash((mapped_key, mapped_value))

        grouper_id_to_grouper_label_mapping[grouper_id] = schemas.Label(
            key=mapped_key, value=mapped_value
        )
        grouper_id_to_label_ids_mapping[grouper_id].append(label.id)

    return {
        "grouper_id_to_label_ids_mapping": grouper_id_to_label_ids_mapping,
        "grouper_id_to_grouper_label_mapping": grouper_id_to_grouper_label_mapping,
    }


def _create_classification_grouper_mappings(
    mapping_dict: dict[tuple[str, ...], tuple[str, ...]],
    labels: list[models.Label],
) -> dict[str, dict]:
    """Create grouper mappings for use when evaluating classifications."""

    # define mappers to connect groupers with labels
    label_value_to_grouper_value = dict()
    grouper_key_to_labels_mapping = defaultdict(lambda: defaultdict(set))
    grouper_key_to_label_keys_mapping = defaultdict(set)

    for label in labels:
        # the grouper should equal the (label.key, label.value) if it wasn't mapped by the user
        grouper_key, grouper_value = mapping_dict.get(
            (label.key, label.value), (label.key, label.value)
        )

        label_value_to_grouper_value[label.value] = grouper_value
        grouper_key_to_label_keys_mapping[grouper_key].add(label.key)
        grouper_key_to_labels_mapping[grouper_key][grouper_value].add(label)

    return {
        "label_value_to_grouper_value": label_value_to_grouper_value,
        "grouper_key_to_labels_mapping": grouper_key_to_labels_mapping,
        "grouper_key_to_label_keys_mapping": grouper_key_to_label_keys_mapping,
    }


def create_grouper_mappings(
    db: Session,
    labels: list,
    label_map: LabelMapType | None,
    evaluation_type: enums.TaskType,
) -> dict[str, dict]:
    """
    Creates a dictionary of mappings that connect each label with a "grouper" (i.e., a unique ID-key-value combination that can represent one or more labels).
    These mappings enable Valor to group multiple labels together using the label_map argument in each evaluation function.

    Parameters
    ----------
    labels : list
        A list of all labels.
    label_map: LabelMapType, optional
        An optional label map to use when grouping labels. If None is passed, this function will still create the appropriate mappings using individual labels.
    evaluation_type : str
        The type of evaluation to create mappings for.

    Returns
    ----------
    dict[str, dict[str | int, any]]
        A dictionary of mappings that are used at evaluation time to group multiple labels together.
    """

    mapping_functions = {
        enums.TaskType.CLASSIFICATION: _create_classification_grouper_mappings,
        enums.TaskType.OBJECT_DETECTION: _create_detection_grouper_mappings,
        enums.TaskType.SEMANTIC_SEGMENTATION: _create_segmentation_grouper_mappings,
    }
    if evaluation_type not in mapping_functions.keys():
        raise KeyError(
            f"evaluation_type must be one of {mapping_functions.keys()}"
        )

    # create a map of labels to groupers; will be empty if the user didn't pass a label_map
    grouper_key_to_value = defaultdict(list)
    mapping_dict = dict()
    if label_map:
        for label, grouper in label_map:
            mapping_dict[tuple(label)] = tuple(grouper)
            grouper_key_to_value[grouper[0]].append(grouper[1])

        # add grouper labels to database (if they don't exist)
        grouper_labels = set(mapping_dict.values())
        existing_labels = {
            (row.key, row.value)
            for row in (
                db.query(models.Label)
                .where(
                    or_(
                        *[
                            and_(
                                models.Label.key == key,
                                models.Label.value.in_(values),
                            )
                            for key, values in grouper_key_to_value.items()
                        ]
                    )
                )
                .all()
            )
        }
        labels_to_create = list(grouper_labels - existing_labels)
        core.create_labels(
            db=db,
            labels=[
                schemas.Label(key=key, value=value)
                for key, value in labels_to_create
            ],
        )

    return mapping_functions[evaluation_type](mapping_dict, labels)


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
        )
        if filters.annotations
        else task_type_condition
    )

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

from collections import defaultdict
from typing import Callable, Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from valor_api import enums, logger, schemas
from valor_api.backend import core, models
from valor_api.backend.query import Query

LabelMapType = list[list[list[str]]]


def _create_detection_grouper_mappings(
    mapping_dict: dict[tuple[str, ...], tuple[str, ...]],
    labels: list[models.Label],
) -> dict[str, dict]:
    """Create grouper mappings for use when evaluating detections."""

    label_id_to_grouper_id_mapping = {}
    grouper_id_to_grouper_label_mapping = {}
    grouper_id_to_label_ids_mapping = defaultdict(list)

    for label in labels:
        mapped_key, mapped_value = mapping_dict.get(
            (label.key, label.value), (label.key, label.value)
        )
        # create an integer to track each group by
        grouper_id = hash((mapped_key, mapped_value))

        label_id_to_grouper_id_mapping[label.id] = grouper_id
        grouper_id_to_grouper_label_mapping[grouper_id] = schemas.Label(
            key=mapped_key, value=mapped_value
        )
        grouper_id_to_label_ids_mapping[grouper_id].append(label.id)

    return {
        "label_id_to_grouper_id_mapping": label_id_to_grouper_id_mapping,
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
    label_value_to_grouper_value = {}
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
    mapping_dict = (
        {tuple(label): tuple(grouper) for label, grouper in label_map}
        if label_map
        else {}
    )

    return mapping_functions[evaluation_type](mapping_dict, labels)


def get_or_create_row(
    db: Session,
    model_class: type,
    mapping: dict,
    columns_to_ignore: list[str] | None = None,
):
    """
    Tries to get the row defined by mapping. If that exists then its mapped object is returned. Otherwise a row is created by `mapping` and the newly created object is returned.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_class : type
        The type of model.
    mapping : dict
        The mapping to use when creating the row.
    columns_to_ignore : List[str]
        Specifies any columns to ignore in forming the WHERE expression. This can be used for numerical columns that might slightly differ but are essentially the same.

    Returns
    ----------
    any
        A model class object.
    """
    columns_to_ignore = columns_to_ignore or []

    # create the query from the mapping
    where_expressions = [
        (getattr(model_class, k) == v)
        for k, v in mapping.items()
        if k not in columns_to_ignore
    ]
    where_expression = where_expressions[0]
    for exp in where_expressions[1:]:
        where_expression = where_expression & exp

    db_element = db.scalar(select(model_class).where(where_expression))

    if not db_element:
        db_element = model_class(**mapping)
        db.add(db_element)
        db.flush()
        db.commit()

    return db_element


def create_metric_mappings(
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
    ],
    evaluation_id: int,
) -> list[dict]:
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

    Returns
    ----------
    List[Dict]
        A list of metric mappings.
    """
    ret = []
    for metric in metrics:
        if hasattr(metric, "label"):
            label = core.fetch_label(
                db=db,
                label=metric.label,  # type: ignore - https://github.com/microsoft/pylance-release/issues/2237
            )

            # create the label in the database if it doesn't exist
            # this is useful if the user maps existing labels to a non-existant grouping label
            if not label:
                label = core.create_labels(db=db, labels=[metric.label])  # type: ignore - https://github.com/microsoft/pylance-release/issues/2237
                label_id = label[0].id
            else:
                label_id = label.id

            ret.append(
                metric.db_mapping(
                    label_id=label_id,  # type: ignore - https://github.com/microsoft/pylance-release/issues/2237
                    evaluation_id=evaluation_id,
                )
            )
        else:
            ret.append(metric.db_mapping(evaluation_id=evaluation_id))  # type: ignore - unnecessary since we're checking for label attribute above

    return ret


def log_evaluation_analytics(
    db: Session,
    evaluation_id: int,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
):
    """
    Store analytics regarding the evaluation's execution in the metadata field of the evaluation table.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    """
    # get ground truth, prediction, annotation, and label counts
    gt_subquery = (
        Query(
            models.GroundTruth,
        )
        .filter(groundtruth_filter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    gts = db.execute(
        select(
            gt_subquery.c.annotation_id,
            gt_subquery.c.label_id,
        ).select_from(gt_subquery)
    ).all()

    # handle edge case where no gts come back
    if not gts:
        gt_annotation_id, gt_label_id = set(), set()
    else:
        gt_annotation_id, gt_label_id = map(set, zip(*gts))

    pd_subquery = (
        Query(
            models.Prediction,
        )
        .filter(prediction_filter)
        .predictions(as_subquery=False)
        .alias()
    )

    pds = db.execute(
        select(
            pd_subquery.c.annotation_id,
            pd_subquery.c.label_id,
        ).select_from(pd_subquery)
    ).all()

    if not pds:
        pd_annotation_id, pd_label_id = set(), set()
    else:
        pd_annotation_id, pd_label_id = map(set, zip(*pds))

    annotation_cnt = len(gt_annotation_id | pd_annotation_id)
    label_cnt = len(gt_label_id | pd_label_id)

    # get datum count
    pd_datums = db.query(
        Query(models.Dataset.name, models.Datum.id)  # type: ignore - sqlalchemy issues
        .filter(prediction_filter)
        .predictions()
    ).all()
    gt_datums = db.query(
        Query(models.Dataset.name, models.Datum.id)  # type: ignore - sqlalchemy issues
        .filter(groundtruth_filter)
        .groundtruths()
    ).all()
    unique_datums = set(pd_datums + gt_datums)
    datum_cnt = len(unique_datums)

    evaluation = core.fetch_evaluation_from_id(db, evaluation_id)

    server_time = db.execute(func.now()).scalar().replace(tzinfo=None)  # type: ignore - guaranteed to return server time if psql is running
    duration = (server_time - evaluation.created_at).total_seconds()

    output = {
        "annotations": annotation_cnt,
        "labels": label_cnt,
        "datums": datum_cnt,
        "duration": duration,
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

from sqlalchemy import Subquery, and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select

from valor_api import schemas
from valor_api.backend import models
from valor_api.backend.query import Query

LabelMapType = list[list[list[str]]]


def validate_matching_label_keys(
    db: Session,
    label_map: LabelMapType | None,
    prediction_filter: schemas.Filter,
    groundtruth_filter: schemas.Filter,
) -> None:
    """
    Validates that every datum has the same set of label keys for both ground truths and predictions. This check is only needed for classification tasks.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction_filter : schemas.Filter
        The filter to be used to query predictions.
    groundtruth_filter : schemas.Filter
        The filter to be used to query groundtruths.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.


    Raises
    -------
    ValueError
        If the distinct ground truth label keys don't match the distinct prediction label keys for any datum.
    """

    gts = (
        Query(
            models.Annotation.datum_id.label("datum_id"),
            models.Label.key.label("label_key"),
            models.Label.value.label("label_value"),
        )
        .filter(groundtruth_filter)
        .groundtruths(as_subquery=False)
        .alias()
    )

    gt_label_keys_by_datum = (
        select(
            gts.c.datum_id,
            func.array_agg(gts.c.label_key + ", " + gts.c.label_value).label(
                "gt_labels"
            ),
        )
        .select_from(gts)
        .group_by(gts.c.datum_id)
        .subquery()
    )

    preds = (
        Query(
            models.Annotation.datum_id.label("datum_id"),
            models.Label.key.label("label_key"),
            models.Label.value.label("label_value"),
        )
        .filter(prediction_filter)
        .predictions(as_subquery=False)
        .alias()
    )

    preds_label_keys_by_datum = (
        select(
            preds.c.datum_id,
            func.array_agg(
                preds.c.label_key + ", " + preds.c.label_value
            ).label("pred_labels"),
        )
        .select_from(preds)
        .group_by(preds.c.datum_id)
        .subquery()
    )

    joined = (
        select(
            preds_label_keys_by_datum.c.datum_id,
            preds_label_keys_by_datum.c.pred_labels,
            gt_label_keys_by_datum.c.gt_labels,
        )
        .select_from(preds_label_keys_by_datum)
        .join(
            gt_label_keys_by_datum,
            gt_label_keys_by_datum.c.datum_id
            == preds_label_keys_by_datum.c.datum_id,
        )
        .subquery()
    )

    # map the keys to the using the label_map if necessary
    label_map_lookup = {}
    if label_map:
        for entry in label_map:
            label_map_lookup[tuple(entry[0])] = tuple(entry[1])

    results = [
        {
            "datum_id": datum_id,
            "pred_keys": set(
                [
                    (
                        label_map_lookup[tuple(entry.split(", "))][0]
                        if tuple(entry.split(", ")) in label_map_lookup
                        else tuple(entry.split(", "))[0]
                    )
                    for entry in pred_labels
                ]
            ),
            "gt_keys": set(
                [
                    (
                        label_map_lookup[tuple(entry.split(", "))][0]
                        if tuple(entry.split(", ")) in label_map_lookup
                        else tuple(entry.split(", "))[0]
                    )
                    for entry in gt_labels
                ]
            ),
        }
        for datum_id, pred_labels, gt_labels in db.query(joined).all()
    ]

    for entry in results:
        if not entry["pred_keys"] == entry["gt_keys"]:
            raise ValueError(
                f"Ground truth label keys must match prediction label keys for classification tasks. Found the following mismatch: {entry}."
            )


def fetch_label(
    db: Session,
    label: schemas.Label,
) -> models.Label | None:
    """
    Fetch label from the database.

    Parameters
    ----------
    db : Session
        SQLAlchemy ORM session.
    label : schemas.Label
        Label schema to search for in the database.

    Returns
    -------
    models.Label | None
    """
    return db.query(
        select(models.Label)
        .where(
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
            )
        )
        .subquery()
    ).one_or_none()


def fetch_matching_labels(
    db: Session,
    labels: list[schemas.Label],
) -> list[models.Label]:
    """
    Fetch matching labels from the database.

    If a label in the search set is not found, no output is generated.

    Parameters
    ----------
    db : Session
        SQLAlchemy ORM session.
    labels : List[schemas.Label]
        List of label schemas to search for in the database.
    """
    return db.query(
        select(models.Label)
        .where(
            or_(
                *[
                    and_(
                        models.Label.key == label.key,
                        models.Label.value == label.value,
                    )
                    for label in labels
                ]
            )
        )
        .subquery()
    ).all()


def create_labels(
    db: Session,
    labels: list[schemas.Label],
) -> list[models.Label]:
    """
    Add a list of labels to create in the database.

    Handles cases where the label already exists in the database.

    The returned list of `models.Label` retains the inputs ordering.

    Parameters
    -------
    db : Session
        The database session to query against.
    labels : list[schemas.Label]
        A list of labels to add to postgis.

    Returns
    -------
    List[models.Label]
        A list of corresponding label rows from the database.
    """
    # check if empty
    if not labels:
        return []

    # remove duplicates
    labels_no_duplicates = list(set(labels))

    # get existing labels
    existing_labels = {
        (label.key, label.value)
        for label in fetch_matching_labels(db, labels_no_duplicates)
    }

    # create new labels
    new_labels_set = {
        (label.key, label.value)
        for label in labels_no_duplicates
        if (label.key, label.value) not in existing_labels
    }
    new_labels = [
        models.Label(key=label[0], value=label[1])
        for label in list(new_labels_set)
    ]

    # upload the labels that were missing
    try:
        db.add_all(new_labels)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e  # this should never be called

    # get existing labels and match output order to users request
    existing_labels = {
        (label.key, label.value): label
        for label in fetch_matching_labels(db, labels_no_duplicates)
    }
    return [existing_labels[(label.key, label.value)] for label in labels]


def _getter_statement(
    selection,
    filters: schemas.Filter | None = None,
    ignore_groundtruths: bool = False,
    ignore_predictions: bool = False,
) -> Subquery | Select:
    """Builds sql statement for other functions."""
    stmt = Query(selection)
    if filters:
        stmt = stmt.filter(filters)
    if not ignore_groundtruths and ignore_predictions:
        stmt = stmt.groundtruths(as_subquery=False)
    elif ignore_groundtruths and not ignore_predictions:
        stmt = stmt.predictions(as_subquery=False)
    else:
        stmt = stmt.any(as_subquery=False)
    return stmt


def get_labels(
    db: Session,
    filters: schemas.Filter | None = None,
    ignore_groundtruths: bool = False,
    ignore_predictions: bool = False,
) -> set[schemas.Label]:
    """
    Returns a set of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type).

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.
    ignore_groundtruths : bool, default=False
        An optional toggle to ignore labels associated with groundtruths.
    ignore_predictions : bool, default=False
        An optional toggle to ignore labels associated with predictions.

    Returns
    ----------
    set[schemas.Label]
        A set of labels.
    """
    stmt = _getter_statement(
        selection=models.Label,
        filters=filters,
        ignore_groundtruths=ignore_groundtruths,
        ignore_predictions=ignore_predictions,
    )
    return {
        schemas.Label(key=label.key, value=label.value)
        for label in db.query(stmt.subquery()).all()  # type: ignore - sqlalchemy typing issue
    }


def get_label_keys(
    db: Session,
    filters: schemas.Filter | None = None,
    ignore_groundtruths: bool = False,
    ignore_predictions: bool = False,
) -> set[str]:
    """
    Returns all unique label keys.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.
    ignore_groundtruths : bool, default=False
        An optional toggle to ignore label keys associated with groundtruths.
    ignore_predictions : bool, default=False
        An optional toggle to ignore label keys associated with predictions.

    Returns
    ----------
    set[str]
        A set of label keys.
    """
    stmt = _getter_statement(
        selection=models.Label.key,
        filters=filters,
        ignore_groundtruths=ignore_groundtruths,
        ignore_predictions=ignore_predictions,
    )
    return {key for key in db.scalars(stmt)}  # type: ignore - sqlalchemy typing issue


def get_joint_labels(
    db: Session,
    lhs: schemas.Filter,
    rhs: schemas.Filter,
) -> list[schemas.Label]:
    """
    Returns all unique labels that are shared between both filters.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    lhs : list[schemas.Filter]
        Filter defining first label set.
    rhs : list[schemas.Filter]
        Filter defining second label set.

    Returns
    ----------
    list[schemas.Label]
        A list of labels.
    """
    lhs_labels = get_labels(db, lhs, ignore_predictions=True)
    rhs_labels = get_labels(db, rhs, ignore_groundtruths=True)
    return list(lhs_labels.intersection(rhs_labels))


def get_joint_keys(
    db: Session,
    lhs: schemas.Filter,
    rhs: schemas.Filter,
) -> list[str]:
    """
    Returns all unique label keys that are shared between both filters.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    lhs : list[schemas.Filter]
        Filter defining first label set.
    rhs : list[schemas.Filter]
        Filter defining second label set.

    Returns
    ----------
    set[schemas.Label]
        A list of labels.
    """
    lhs_keys = get_label_keys(db, lhs, ignore_predictions=True)
    rhs_keys = get_label_keys(db, rhs, ignore_groundtruths=True)
    return list(lhs_keys.intersection(rhs_keys))


def get_disjoint_labels(
    db: Session,
    lhs: schemas.Filter,
    rhs: schemas.Filter,
    label_map: LabelMapType | None = None,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Returns all unique labels that are not shared between both filters.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    lhs : list[schemas.Filter]
        Filter defining first label set.
    rhs : list[schemas.Filter]
        Filter defining second label set.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

    Returns
    ----------
    Tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of disjoint labels, where the first element is those labels which are present in lhs label set but absent in rhs label set.
    """
    lhs_labels = get_labels(db, lhs, ignore_predictions=True)
    rhs_labels = get_labels(db, rhs, ignore_groundtruths=True)

    # don't count user-mapped labels as disjoint
    mapped_labels = set()
    if label_map:
        for map_from, map_to in label_map:
            mapped_labels.add(
                schemas.Label(key=map_from[0], value=map_from[1])
            )
            mapped_labels.add(schemas.Label(key=map_to[0], value=map_to[1]))

    lhs_unique = list(lhs_labels - rhs_labels - mapped_labels)
    rhs_unique = list(rhs_labels - lhs_labels - mapped_labels)
    return (lhs_unique, rhs_unique)


def get_disjoint_keys(
    db: Session,
    lhs: schemas.Filter,
    rhs: schemas.Filter,
    label_map: LabelMapType | None = None,
) -> tuple[list[str], list[str]]:
    """
    Returns all unique label keys that are not shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    lhs : list[schemas.Filter]
        Filter defining first label set.
    rhs : list[schemas.Filter]
        Filter defining second label set.
    label_map: LabelMapType, optional,

        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

    Returns
    ----------
    Tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of disjoint label key, where the first element is those labels which are present in lhs but absent in rhs.
    """
    lhs_keys = get_label_keys(db, lhs, ignore_predictions=True)
    rhs_keys = get_label_keys(db, rhs, ignore_groundtruths=True)

    # don't count user-mapped labels as disjoint
    mapped_keys = set()
    if label_map:
        for map_from, map_to in label_map:
            mapped_keys.add(map_from[0])
            mapped_keys.add(map_to[0])

    lhs_unique = list(lhs_keys - rhs_keys - mapped_keys)
    rhs_unique = list(rhs_keys - lhs_keys - mapped_keys)
    return (lhs_unique, rhs_unique)


def fetch_labels(
    db: Session,
    filter_: schemas.Filter,
    ignore_groundtruths: bool = False,
    ignore_predictions: bool = False,
) -> set[models.Label]:
    """
    Fetch a set of models.Label entries from the database.

    Parameters
    ----------
    db : Session
        SQLAlchemy ORM session.
    filter_ : schemas.Filter
        Filter to constrain results by.

    Returns
    -------
    set[models.Label]
    """
    stmt = _getter_statement(
        selection=models.Label,
        filters=filter_,
        ignore_groundtruths=ignore_groundtruths,
        ignore_predictions=ignore_predictions,
    )
    return set(db.query(stmt.subquery()).all())  # type: ignore - sqlalchemy typing issue


def fetch_union_of_labels(
    db: Session,
    lhs: schemas.Filter,
    rhs: schemas.Filter,
) -> list[models.Label]:
    """
    Returns a list of unique models.Label that are shared between both filters.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    lhs : list[schemas.Filter]
        Filter defining first label set.
    rhs : list[schemas.Filter]
        Filter defining second label set.

    Returns
    ----------
    list[models.Label]
        A list of labels.
    """
    lhs_labels = fetch_labels(db, filter_=lhs, ignore_predictions=True)
    rhs_labels = fetch_labels(db, filter_=rhs, ignore_groundtruths=True)
    return list(lhs_labels.union(rhs_labels))

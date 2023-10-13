from sqlalchemy import select
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core


def get_or_create_row(
    db: Session,
    model_class: type,
    mapping: dict,
    columns_to_ignore: list[str] = None,
) -> any:
    """Tries to get the row defined by mapping. If that exists then
    its mapped object is returned. Otherwise a row is created by `mapping` and the newly created
    object is returned. `columns_to_ignore` specifies any columns to ignore in forming the where
    expression. this can be used for numerical columns that might slightly differ but are essentially the same
    (and where the other columns serve as unique identifiers)
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
    metrics: list[
        schemas.APMetric
        | schemas.APMetricAveragedOverIOUs
        | schemas.mAPMetric
        | schemas.mAPMetricAveragedOverIOUs
    ],
    evaluation_id: int,
) -> list[dict]:

    labels = set(
        [
            (metric.label.key, metric.label.value)
            for metric in metrics
            if hasattr(metric, "label")
        ]
    )
    label_map = {
        (label[0], label[1]): core.get_label(
            db, label=schemas.Label(key=label[0], value=label[1])
        ).id
        for label in labels
    }

    ret = []
    for metric in metrics:
        if hasattr(metric, "label"):
            ret.append(
                metric.db_mapping(
                    label_id=label_map[(metric.label.key, metric.label.value)],
                    evaluation_id=evaluation_id,
                )
            )
        else:
            ret.append(metric.db_mapping(evaluation_id=evaluation_id))

    return ret

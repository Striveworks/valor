from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Query, Session

from valor_api.backend.models import Annotation, GroundTruth, Prediction
from valor_api.backend.query.solvers import solver
from valor_api.backend.query.types import LabelSourceAlias
from valor_api.schemas.filters import Filter


def generate_select(
    *args: Any,
    filter_: Filter | None = None,
    label_source: LabelSourceAlias = Annotation,
) -> Select[Any]:
    """
    Creates a select statement from provided arguments and filters.

    The label source determines which graph structure to use.

    Parameters
    ----------
    *args : Any
        A variable list of models or model attributes. (e.g. Label or Label.key)
    filter_ : Filter  optional
        An optional filter.
    label_source : LabelSourceAlias, default=Annotation
        The table to source labels from. This determines graph structure.

    Returns
    -------
    Select[Any]
        A select statement that meets all conditions.

    Raises
    ------
    ValueError
        If label source is not a valid table.
    RunTimeError
        If the output of the solver does not match the input type.
    """
    if label_source not in {Annotation, GroundTruth, Prediction}:
        raise ValueError(
            "Label source must be either Annotation, GroundTruth or Prediction."
        )
    query = solver(
        *args,
        stmt=select(*args),
        filter_=filter_,
        label_source=label_source,
    )
    if not isinstance(query, Select):
        raise RuntimeError(
            "The output type of 'generate_query' should match the type of the 'select_statement' arguement."
        )
    return query


def generate_query(
    *args: Any,
    db: Session,
    filter_: Filter | None = None,
    label_source: LabelSourceAlias = Annotation,
) -> Query[Any]:
    """
    Creates a query statement from provided arguments and filters.

    The label source determines which graph structure to use.

    Parameters
    ----------
    *args : Any
        A variable list of models or model attributes. (e.g. Label or Label.key)
    db : Session
        The database session to call query against.
    filter_ : Filter  optional
        An optional filter.
    label_source : LabelSourceAlias, default=Annotation
        The table to source labels from. This determines graph structure.

    Returns
    -------
    Select[Any]
        A select statement that meets all conditions.

    Raises
    ------
    ValueError
        If label source is not a valid table.
    RunTimeError
        If the output of the solver does not match the input type.
    """
    if label_source not in {Annotation, GroundTruth, Prediction}:
        raise ValueError(
            "Label source must be either Annotation, GroundTruth or Prediction."
        )
    query = solver(
        *args,
        stmt=db.query(*args),
        filter_=filter_,
        label_source=label_source,
    )
    if not isinstance(query, Query):
        raise RuntimeError(
            "The output type of 'generate_query' should match the type of the 'select_statement' arguement."
        )
    return query

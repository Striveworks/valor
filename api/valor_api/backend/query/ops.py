from typing import Any

from sqlalchemy import Select, select

from valor_api.backend.models import Annotation, GroundTruth, Prediction
from valor_api.backend.query.solvers import solver
from valor_api.schemas.filters import AdvancedFilter, Filter


def select_from_annotations(
    *args, filter_: AdvancedFilter | None = None
) -> Select[Any]:
    """TODO"""
    query = solver(
        *args, stmt=select(*args), filter_=filter_, label_source=Annotation
    )
    if not isinstance(query, Select):
        raise RuntimeError(
            "The output type of 'generate_query' should match the type of the 'select_statement' arguement."
        )
    return query


def select_from_groundtruths(
    *args, filter_: AdvancedFilter | None = None
) -> Select[Any]:
    """TODO"""
    query = solver(
        *args, stmt=select(*args), filter_=filter_, label_source=GroundTruth
    )
    if not isinstance(query, Select):
        raise RuntimeError(
            "The output type of 'generate_query' should match the type of the 'select_statement' arguement."
        )
    return query


def select_from_predictions(
    *args, filter_: AdvancedFilter | None = None
) -> Select[Any]:
    """TODO"""
    query = solver(
        *args, stmt=select(*args), filter_=filter_, label_source=Prediction
    )
    if not isinstance(query, Select):
        raise RuntimeError(
            "The output type of 'generate_query' should match the type of the 'select_statement' arguement."
        )
    return query


class Query:
    """
    Query generator object.

    Attributes
    ----------
    *args : TableTypeAlias | InstrumentedAttribute
        args is a list of models or model attributes. (e.g. Label or Label.key)

    Examples
    ----------
    Querying
    >>> f = schemas.Filter(...)
    >>> q = Query(Label).filter(f).any()

    Querying model attributes.
    >>> f = schemas.Filter(...)
    >>> q = Query(Label.key).filter(f).any()
    """

    def __init__(self, *args):
        self._args = args
        self._filter = None

    def select_from(self, *args):
        return self

    def filter(self, filter_: Filter | None):
        """Parses `schemas.Filter`"""
        if filter_ is None:
            return self
        if not isinstance(filter_, Filter):
            raise TypeError(
                "filter_ should be of type `schemas.Filter` or `None`"
            )
        self._filter = filter_
        return self

    def any(
        self,
        name: str = "generated_subquery",
        *,
        as_subquery: bool = True,
    ):
        """
        Generates a sqlalchemy subquery. Graph is chosen automatically as best fit.
        """
        filter_ = (
            AdvancedFilter.from_simple_filter(self._filter)
            if self._filter
            else None
        )
        stmt = select_from_annotations(*self._args, filter_=filter_)
        if as_subquery:
            return stmt.subquery(name)
        else:
            return stmt

    def groundtruths(
        self, name: str = "generated_subquery", *, as_subquery: bool = True
    ):
        """
        Generates a sqlalchemy subquery using a groundtruths-focused graph.
        """
        filter_ = (
            AdvancedFilter.from_simple_filter(
                self._filter, ignore_predictions=True
            )
            if self._filter
            else None
        )
        stmt = select_from_groundtruths(*self._args, filter_=filter_)
        if as_subquery:
            return stmt.subquery(name)
        else:
            return stmt

    def predictions(
        self,
        name: str = "generated_subquery",
        *,
        as_subquery: bool = True,
    ):
        """
        Generates a sqlalchemy subquery using a predictions-focused graph.
        """
        filter_ = (
            AdvancedFilter.from_simple_filter(
                self._filter, ignore_groundtruths=True
            )
            if self._filter
            else None
        )
        stmt = select_from_predictions(*self._args, filter_=filter_)
        if as_subquery:
            return stmt.subquery(name)
        else:
            return stmt

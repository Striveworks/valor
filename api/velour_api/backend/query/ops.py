from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.elements import BinaryExpression, ColumnElement

from velour_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from velour_api.backend.query.filtering import (
    filter_by_annotation,
    filter_by_dataset,
    filter_by_datum,
    filter_by_label,
    filter_by_model,
    filter_by_prediction,
)
from velour_api.backend.query.mapping import map_arguments_to_tables
from velour_api.backend.query.solvers import solve_graph
from velour_api.schemas import Filter


class Query:
    """
    Query generator object.

    Attributes
    ----------
    *args : DeclarativeMeta | InstrumentedAttribute
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
        self._selected: set[DeclarativeMeta] = map_arguments_to_tables(args)
        self._filtered = set()
        self._expressions: dict[
            DeclarativeMeta, list[ColumnElement[bool]]
        ] = {}

    def select_from(self, *args):
        self._selected = map_arguments_to_tables(args)
        return self

    def _add_expressions(
        self,
        table: DeclarativeMeta,
        expressions: list[ColumnElement[bool] | BinaryExpression],
    ) -> None:
        if len(expressions) == 0:
            return
        self._filtered.add(table)
        if table not in self._expressions:
            self._expressions[table] = []
        self._expressions[table].extend(expressions)

    def filter(self, filter_: Filter | None):  # type: ignore - method "filter" overrides class "Query" in an incompatible manner
        """Parses `schemas.Filter`"""
        if filter_ is None:
            return self
        if not isinstance(filter_, Filter):
            raise TypeError(
                "filter_ should be of type `schemas.Filter` or `None`"
            )
        self._add_expressions(Annotation, filter_by_annotation(filter_))
        self._add_expressions(Model, filter_by_model(filter_))
        self._add_expressions(Label, filter_by_label(filter_))
        self._add_expressions(Dataset, filter_by_dataset(filter_))
        self._add_expressions(Datum, filter_by_datum(filter_))
        self._add_expressions(Prediction, filter_by_prediction(filter_))
        return self

    def any(
        self,
        name: str = "generated_subquery",
        *,
        pivot: DeclarativeMeta | None = None,
        as_subquery: bool = True,
    ):
        """
        Generates a sqlalchemy subquery. Graph is chosen automatically as best fit.
        """
        query, subquery = solve_graph(
            select_args=self._args,
            selected_tables=self._selected,
            filter_by_tables=self._filtered,
            expressions=self._expressions,
            pivot_table=pivot,
        )
        if query is None:
            raise RuntimeError("No solution found to query.")

        if subquery is not None:
            query = query.where(Datum.id.in_(subquery))
        return query.subquery(name) if as_subquery else query

    def groundtruths(
        self, name: str = "generated_subquery", *, as_subquery: bool = True
    ):
        """
        Generates a sqlalchemy subquery using a groundtruths-focused graph.
        """
        return self.any(name, pivot=GroundTruth, as_subquery=as_subquery)

    def predictions(
        self,
        name: str = "generated_subquery",
        *,
        as_subquery: bool = True,
    ):
        """
        Generates a sqlalchemy subquery using a predictions-focused graph.
        """
        return self.any(name, pivot=Prediction, as_subquery=as_subquery)

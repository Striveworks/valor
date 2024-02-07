from sqlalchemy.orm.attributes import InstrumentedAttribute
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
        self._selected: set[DeclarativeMeta | None] = set(
            [
                self._map_attribute_to_table(argument)
                for argument in args
                if (
                    isinstance(argument, DeclarativeMeta)
                    or isinstance(argument, InstrumentedAttribute)
                    or hasattr(argument, "__visit_name__")
                )
            ]
        )
        self._filtered = set()
        self._excluded = set()
        self._expressions: dict[
            DeclarativeMeta, list[ColumnElement[bool]]
        ] = {}

    def select_from(self, *args):
        self._selected: set[DeclarativeMeta | None] = set(
            [
                self._map_attribute_to_table(argument)
                for argument in args
                if (
                    isinstance(argument, DeclarativeMeta)
                    or isinstance(argument, InstrumentedAttribute)
                    or hasattr(argument, "__visit_name__")
                )
            ]
        )
        return self

    def exclude(self, *args):
        """Exclude tables from query."""
        for table in args:
            if isinstance(table, DeclarativeMeta):
                self._excluded.add(table)
        return self

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

    def compile(
        self,
        name: str = "generated_subquery",
        *,
        pivot: DeclarativeMeta | None = None,
        as_subquery: bool = True,
    ):
        """
        Generates a sqlalchemy subquery. Graph is chosen automatically as best fit.
        """
        if self._selected is not None:
            raise RuntimeError("No tables selected.")

        self._selected = self._selected - self._excluded
        self._filtered = self._filtered - self._excluded
        query, subquery = solve_graph(
            select_args=self._args,  # type: ignore
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
        return self.compile(name, pivot=GroundTruth, as_subquery=as_subquery)

    def predictions(
        self,
        name: str = "generated_subquery",
        *,
        as_subquery: bool = True,
    ):
        """
        Generates a sqlalchemy subquery using a predictions-focused graph.
        """
        return self.compile(name, pivot=Prediction, as_subquery=as_subquery)

    def _map_attribute_to_table(
        self, attr: InstrumentedAttribute | DeclarativeMeta
    ) -> DeclarativeMeta | None:
        if isinstance(attr, DeclarativeMeta):
            return attr
        elif isinstance(attr, InstrumentedAttribute):
            table_name = attr.table.name
        elif hasattr(attr, "__visit_name__"):
            table_name = attr.__visit_name__
        else:
            table_name = None

        match table_name:
            case Dataset.__tablename__:
                return Dataset
            case Model.__tablename__:
                return Model
            case Datum.__tablename__:
                return Datum
            case Annotation.__tablename__:
                return Annotation
            case GroundTruth.__tablename__:
                return GroundTruth
            case Prediction.__tablename__:
                return Prediction
            case Label.__tablename__:
                return Label
            case _:
                return None

    def _add_expressions(
        self, table, expressions: list[ColumnElement[bool] | BinaryExpression]
    ) -> None:
        self._filtered.add(table)
        if table not in self._expressions:
            self._expressions[table] = []
        self._expressions[table].extend(expressions)

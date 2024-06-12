import operator

from geoalchemy2.functions import ST_Area, ST_Count, ST_GeomFromGeoJSON
from sqlalchemy import (
    CTE,
    TIMESTAMP,
    BinaryExpression,
    Boolean,
    Float,
    Integer,
    and_,
    cast,
    func,
    not_,
    or_,
    select,
)
from sqlalchemy.dialects.postgresql import INTERVAL, TEXT

from valor_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    Embedding,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor_api.backend.query.types import LabelSourceAlias, TableTypeAlias
from valor_api.schemas.filters import (
    Condition,
    Filter,
    FilterOperator,
    LogicalFunction,
    SupportedType,
    Symbol,
    Value,
)
from valor_api.schemas.geometry import (
    Box,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


def raise_not_implemented(x):
    raise NotImplementedError(f"{x} is not implemented.")


# Map an operation to a callable function.
map_opstr_to_operator = {
    FilterOperator.EQ: operator.eq,
    FilterOperator.NE: operator.ne,
    FilterOperator.GT: operator.gt,
    FilterOperator.GTE: operator.ge,
    FilterOperator.LT: operator.lt,
    FilterOperator.LTE: operator.le,
    FilterOperator.INTERSECTS: lambda lhs, rhs: func.ST_Intersects(lhs, rhs),
    FilterOperator.INSIDE: lambda lhs, rhs: func.ST_Covers(rhs, lhs),
    FilterOperator.OUTSIDE: lambda lhs, rhs: not_(func.ST_Covers(rhs, lhs)),
    FilterOperator.ISNULL: lambda lhs, _: lhs.is_(None),
    FilterOperator.ISNOTNULL: lambda lhs, _: lhs.isnot(None),
    FilterOperator.CONTAINS: lambda lhs, rhs: lhs.op("?")(rhs),
}


# Map a symbol to a tuple containing (table, column, type string).
map_symbol_to_resources = {
    Symbol.DATASET_NAME: (Dataset, Dataset.name),
    Symbol.MODEL_NAME: (Model, Model.name),
    Symbol.DATUM_UID: (Datum, Datum.uid),
    Symbol.BOX: (Annotation, Annotation.box),
    Symbol.POLYGON: (Annotation, Annotation.polygon),
    Symbol.RASTER: (Annotation, Annotation.raster),
    Symbol.TASK_TYPE: (Annotation, Annotation.implied_task_types),
    Symbol.EMBEDDING: (Embedding, Embedding.value),
    Symbol.LABELS: (Label, Label),
    Symbol.LABEL_KEY: (Label, Label.key),
    Symbol.LABEL_VALUE: (Label, Label.value),
    Symbol.SCORE: (Prediction, Prediction.score),
    # 'area' attribute
    Symbol.BOX_AREA: (Annotation, ST_Area(Annotation.box)),
    Symbol.POLYGON_AREA: (Annotation, ST_Area(Annotation.polygon)),
    Symbol.RASTER_AREA: (Annotation, ST_Count(Annotation.raster)),
    # backend use only
    Symbol.DATASET_ID: (Dataset, Dataset.id),
    Symbol.MODEL_ID: (Model, Model.id),
    Symbol.DATUM_ID: (Datum, Datum.id),
    Symbol.ANNOTATION_ID: (Annotation, Annotation.id),
    Symbol.GROUNDTRUTH_ID: (GroundTruth, GroundTruth.id),
    Symbol.PREDICTION_ID: (Prediction, Prediction.id),
    Symbol.LABEL_ID: (Label, Label.id),
    Symbol.EMBEDDING_ID: (Embedding, Embedding.id),
}


# Map a keyed symbol to a tuple containing (table, column, type string).
map_keyed_symbol_to_resources = {
    Symbol.DATASET_META: (Dataset, lambda key: Dataset.meta[key]),
    Symbol.MODEL_META: (Model, lambda key: Model.meta[key]),
    Symbol.DATUM_META: (Datum, lambda key: Datum.meta[key]),
    Symbol.ANNOTATION_META: (Annotation, lambda key: Annotation.meta[key]),
    # 'area' attribute
    Symbol.DATASET_META_AREA: (
        Dataset,
        lambda key: ST_Area(ST_GeomFromGeoJSON(Dataset.meta[key]["value"])),
    ),
    Symbol.MODEL_META_AREA: (
        Model,
        lambda key: ST_Area(ST_GeomFromGeoJSON(Model.meta[key]["value"])),
    ),
    Symbol.DATUM_META_AREA: (
        Datum,
        lambda key: ST_Area(ST_GeomFromGeoJSON(Datum.meta[key]["value"])),
    ),
    Symbol.ANNOTATION_META_AREA: (
        Annotation,
        lambda key: ST_Area(ST_GeomFromGeoJSON(Annotation.meta[key]["value"])),
    ),
}


# Map a type to a type casting function. This is used for accessing JSONB values.
map_type_to_jsonb_type_cast = {
    SupportedType.BOOLEAN: lambda x: x.astext.cast(Boolean),
    SupportedType.INTEGER: lambda x: x.astext.cast(Integer),
    SupportedType.FLOAT: lambda x: x.astext.cast(Float),
    SupportedType.STRING: lambda x: x.astext,
    SupportedType.TASK_TYPE: lambda x: x.astext,
    SupportedType.DATETIME: lambda x: cast(
        x["value"].astext, type_=TIMESTAMP(timezone=True)
    ),
    SupportedType.DATE: lambda x: cast(
        x["value"].astext, type_=TIMESTAMP(timezone=True)
    ),
    SupportedType.TIME: lambda x: cast(x["value"].astext, type_=INTERVAL),
    SupportedType.DURATION: lambda x: cast(x["value"].astext, type_=INTERVAL),
    SupportedType.POINT: lambda x: ST_GeomFromGeoJSON(x["value"]),
    SupportedType.MULTIPOINT: lambda x: ST_GeomFromGeoJSON(x["value"]),
    SupportedType.LINESTRING: lambda x: ST_GeomFromGeoJSON(x["value"]),
    SupportedType.MULTILINESTRING: lambda x: ST_GeomFromGeoJSON(x["value"]),
    SupportedType.POLYGON: lambda x: ST_GeomFromGeoJSON(x["value"]),
    SupportedType.BOX: lambda x: ST_GeomFromGeoJSON(x["value"]),
    SupportedType.MULTIPOLYGON: lambda x: ST_GeomFromGeoJSON(x["value"]),
    SupportedType.GEOJSON: lambda x: ST_GeomFromGeoJSON(x["value"]),
    # unsupported
    SupportedType.RASTER: raise_not_implemented,
    SupportedType.EMBEDDING: raise_not_implemented,
    SupportedType.LABEL: raise_not_implemented,
}


# Map a value type to a type casting function.
map_type_to_type_cast = {
    SupportedType.BOOLEAN: lambda x: x,
    SupportedType.INTEGER: lambda x: x,
    SupportedType.FLOAT: lambda x: x,
    SupportedType.STRING: lambda x: x,
    SupportedType.TASK_TYPE: lambda x: x,
    SupportedType.DATETIME: lambda x: cast(x, type_=TIMESTAMP(timezone=True)),
    SupportedType.DATE: lambda x: cast(x, type_=TIMESTAMP(timezone=True)),
    SupportedType.TIME: lambda x: cast(x, type_=INTERVAL),
    SupportedType.DURATION: lambda x: cast(cast(x, TEXT), type_=INTERVAL),
    SupportedType.POINT: lambda x: ST_GeomFromGeoJSON(
        Point(value=x).to_json()
    ),
    SupportedType.MULTIPOINT: lambda x: ST_GeomFromGeoJSON(
        MultiPoint(value=x).to_json()
    ),
    SupportedType.LINESTRING: lambda x: ST_GeomFromGeoJSON(
        LineString(value=x).to_json()
    ),
    SupportedType.MULTILINESTRING: lambda x: ST_GeomFromGeoJSON(
        MultiLineString(value=x).to_dict()
    ),
    SupportedType.POLYGON: lambda x: ST_GeomFromGeoJSON(
        Polygon(value=x).to_json()
    ),
    SupportedType.BOX: lambda x: ST_GeomFromGeoJSON(Box(value=x).to_json()),
    SupportedType.MULTIPOLYGON: lambda x: ST_GeomFromGeoJSON(
        MultiPolygon(value=x).to_json()
    ),
    SupportedType.GEOJSON: lambda x: ST_GeomFromGeoJSON(x),
    # unsupported
    SupportedType.RASTER: raise_not_implemented,
    SupportedType.EMBEDDING: raise_not_implemented,
    SupportedType.LABEL: raise_not_implemented,
}


def create_cte(condition: Condition) -> tuple[TableTypeAlias, CTE]:
    """
    Creates a CTE from a binary expression.

    Parameters
    ----------
    opstr : FilterOperator
        The expression operator.
    symbol : Symbol
        The lhs of the expression.
    value : Value, optional
        The rhs of the expression, if it exists.

    Returns
    -------
    tuple[TableTypeAlias, CTE]
        A tuple of a table to join on and the CTE.

    Raises
    ------
    NotImplementedError
        If the symbol is not implemented.
    ValueError
        If there is a type mismatch.
    """

    # convert lhs (symbol) to sql representation
    if condition.lhs in map_symbol_to_resources:
        table, lhs = map_symbol_to_resources[condition.lhs]
    elif condition.lhs in map_keyed_symbol_to_resources and condition.lhs_key:
        table, generate_column = map_keyed_symbol_to_resources[condition.lhs]
        lhs = generate_column(condition.lhs_key)
    else:
        raise NotImplementedError(
            f"Symbol '{condition.lhs}' does not match any existing templates."
        )

    if condition.rhs and condition.lhs_key and condition.lhs.type is None:
        lhs = map_type_to_jsonb_type_cast[condition.rhs.type](lhs)
    elif (
        isinstance(condition.rhs, Value)
        and condition.rhs.type != condition.lhs.type
    ):
        raise TypeError(
            f"Type mismatch between '{condition.lhs}' and '{condition.rhs}'."
        )

    op = map_opstr_to_operator[condition.op]
    rhs = (
        map_type_to_type_cast[condition.rhs.type](condition.rhs.value)
        if isinstance(condition.rhs, Value)
        else None
    )

    return (table, select(table.id).where(op(lhs, rhs)).cte())


def _recursive_search_logic_tree(
    func: Condition | LogicalFunction,
    cte_list: list | None = None,
    tables: list[TableTypeAlias] | None = None,
) -> tuple[int | dict, list[CTE], list[TableTypeAlias]]:
    """
    Walks the filtering function to produce dependencies.
    """
    if not isinstance(func, (Condition, LogicalFunction)):
        raise TypeError(
            f"Expected input to be of type 'OneArgFunction | TwoArgFunction | NArgFunction'. Received '{func}'."
        )
    cte_list = cte_list if cte_list else list()
    tables = tables if tables else list()
    logical_tree = dict()

    if isinstance(func, Condition):
        table, cte = create_cte(func)
        tables.append(table)
        cte_list.append(cte)
        return (len(cte_list) - 1, cte_list, tables)
    elif isinstance(func, LogicalFunction):
        if isinstance(func.args, (Condition, LogicalFunction)):
            branch, cte_list, tables = _recursive_search_logic_tree(
                func.args, cte_list, tables
            )
            logical_tree[func.op] = branch
            return (logical_tree, cte_list, tables)
        else:
            branches = list()
            for arg in func.args:
                branch, cte_list, tables = _recursive_search_logic_tree(
                    arg, cte_list, tables
                )
                branches.append(branch)
            logical_tree[func.op] = branches
            return (logical_tree, cte_list, tables)
    else:
        raise TypeError(
            f"Recieved an unsupported type '{type(func)}' in func."
        )


def map_filter_to_tables(
    filters: Filter | None, label_source: LabelSourceAlias
) -> set[TableTypeAlias]:
    """
    Maps a filter to a set of required tables.

    Parameters
    ----------
    filters : Filter
        The filter to search.
    label_source : LabelSourceAlias
        The table to use as a source of labels.

    Returns
    -------
    set[TableTypeAlias]
        The set of tables required by the filter.
    """
    tables = set()
    if filters is not None:
        if filters.datasets:
            tables.add(Dataset)
        if filters.models:
            tables.add(Model)
        if filters.datums:
            tables.add(Datum)
        if filters.annotations:
            tables.add(Annotation)
        if filters.groundtruths:
            table = GroundTruth if label_source is not Prediction else Datum
            tables.add(table)
        if filters.predictions:
            table = Prediction if label_source is not GroundTruth else Datum
            tables.add(table)
        if filters.labels:
            tables.add(Label)
        if filters.embeddings:
            tables.add(Embedding)
    return tables


def generate_dependencies(
    func: Condition | LogicalFunction | None,
) -> tuple[int | dict | None, list[CTE], list[TableTypeAlias]]:
    """
    Recursively generates the dependencies for creating a filter subquery.

    Parameters
    ----------
    func : Condition | LogicalFunction, optional
        An optional filtering function.

    Returns
    -------
    tuple[int | dict | None, list[CTE], list[TableTypeAlias]]
        A tuple containing a logical index tree, ordered list of CTE's and an ordered list of tables.
    """
    if func is None:
        return (None, list(), list())
    return _recursive_search_logic_tree(func)


def generate_logical_expression(
    root: CTE, tree: int | dict[str, int | dict | list], prefix: str
) -> BinaryExpression:
    """
    Generates the 'where' expression from a logical tree.

    Parameters
    ----------
    root : CTE
        The CTE that evaluates the binary expressions.
    tree : int | dict[str, int | dict | list]
        The logical index tree.
    prefix : str
        The prefix of the relevant CTE.

    Returns
    -------
    BinaryExpression
        A binary expression that can be used in a WHERE statement.
    """
    if isinstance(tree, int):
        return getattr(root.c, f"{prefix}{tree}") == 1
    if not isinstance(tree, dict) or len(tree.keys()) != 1:
        raise ValueError("If not an 'int', expected tree to be dictionary.")

    logical_operators = {
        "and": and_,
        "or": or_,
        "not": not_,
    }
    op = list(tree.keys())[0]
    if op == "and" or op == "or":
        args = tree[op]
        if not isinstance(args, list):
            raise ValueError("Expected a list of expressions.")
        return logical_operators[op](
            *[
                (getattr(root.c, f"{prefix}{arg}") == 1)
                if isinstance(arg, int)
                else generate_logical_expression(
                    root=root, tree=arg, prefix=prefix
                )
                for arg in args
            ]
        )
    elif op == "not":
        arg = tree["not"]
        if isinstance(arg, list):
            raise ValueError
        return (
            (getattr(root.c, f"{prefix}{arg}") == 0)
            if isinstance(arg, int)
            else not_(
                generate_logical_expression(root=root, tree=arg, prefix=prefix)
            )
        )
    else:
        raise ValueError

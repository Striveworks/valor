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
    Filter,
    NArgFunction,
    OneArgFunction,
    Symbol,
    TwoArgFunction,
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

# Map an operation to a callable function.
map_opstr_to_operator = {
    "equal": operator.eq,
    "notequal": operator.ne,
    "greaterthan": operator.gt,
    "greaterthanequal": operator.ge,
    "lessthan": operator.lt,
    "lessthanequal": operator.le,
    "intersects": lambda lhs, rhs: func.ST_Intersects(lhs, rhs),
    "inside": lambda lhs, rhs: func.ST_Covers(rhs, lhs),
    "outside": lambda lhs, rhs: not_(func.ST_Covers(rhs, lhs)),
    "isnull": lambda lhs, _: lhs.is_(None),
    "isnotnull": lambda lhs, _: lhs.isnot(None),
    "contains": lambda lhs, rhs: lhs.op("?")(rhs),
}

# Map a symbol to a tuple containing a table and the relevant attribute.
map_name_to_table_column = {
    "dataset.name": (Dataset, Dataset.name),
    "dataset.metadata": (Dataset, Dataset.meta),
    "model.name": (Model, Model.name),
    "model.metadata": (Model, Model.meta),
    "datum.uid": (Datum, Datum.uid),
    "datum.metadata": (Datum, Datum.meta),
    "annotation.bounding_box": (Annotation, Annotation.box),
    "annotation.polygon": (Annotation, Annotation.polygon),
    "annotation.raster": (Annotation, Annotation.raster),
    "annotation.metadata": (Annotation, Annotation.meta),
    "annotation.task_type": (Annotation, Annotation.implied_task_types),
    "annotation.embedding": (Embedding, Embedding.value),
    "annotation.labels": (Label, Label),
    "annotation.model_id": (Annotation, Annotation.model_id),  # remove
    "label.key": (Label, Label.key),
    "label.value": (Label, Label.value),
    "label.score": (Prediction, Prediction.score),
    "label.id": (Label, Label.id),  # remove
    "model.id": (Model, Model.id),
}


# Map a symbol's attribute type to a modifying function.
map_attribute_to_type_cast = {
    "area": {
        "annotation.bounding_box": lambda x: ST_Area(x),
        "annotation.polygon": lambda x: ST_Area(x),
        "annotation.raster": lambda x: ST_Count(x),
        "dataset.metadata": lambda x: ST_Area(x),
        "model.metadata": lambda x: ST_Area(x),
        "datum.metadata": lambda x: ST_Area(x),
        "annotation.metadata": lambda x: ST_Area(x),
    }
}


# Map a symbol's attribute to the type expected by the operation.
map_symbol_attribute_to_type = {"area": "float"}


# Map a type to a type casting function. This is used for accessing JSONB values.
map_type_to_jsonb_type_cast = {
    "boolean": lambda x: x.astext.cast(Boolean),
    "integer": lambda x: x.astext.cast(Integer),
    "float": lambda x: x.astext.cast(Float),
    "string": lambda x: x.astext,
    "tasktype": lambda x: x.astext,
    "datetime": lambda x: cast(
        x["value"].astext, type_=TIMESTAMP(timezone=True)
    ),
    "date": lambda x: cast(x["value"].astext, type_=TIMESTAMP(timezone=True)),
    "time": lambda x: cast(x["value"].astext, type_=INTERVAL),
    "duration": lambda x: cast(x["value"].astext, type_=INTERVAL),
    "point": lambda x: ST_GeomFromGeoJSON(x["value"]),
    "multipoint": lambda x: ST_GeomFromGeoJSON(x["value"]),
    "linestring": lambda x: ST_GeomFromGeoJSON(x["value"]),
    "multilinestring": lambda x: ST_GeomFromGeoJSON(x["value"]),
    "polygon": lambda x: ST_GeomFromGeoJSON(x["value"]),
    "box": lambda x: ST_GeomFromGeoJSON(x["value"]),
    "multipolygon": lambda x: ST_GeomFromGeoJSON(x["value"]),
    "geojson": lambda x: ST_GeomFromGeoJSON(x["value"]),
}


# Map an attribute to a type casting function. This is used for accessing JSONB values.
map_attribute_to_jsonb_type_cast = {
    "area": lambda x: ST_GeomFromGeoJSON(x["value"]),
}


# Map a value type to a type casting function.
map_value_type_to_type_cast = {
    "boolean": lambda x: x,
    "integer": lambda x: x,
    "float": lambda x: x,
    "string": lambda x: x,
    "tasktype": lambda x: x,
    "datetime": lambda x: cast(x, type_=TIMESTAMP(timezone=True)),
    "date": lambda x: cast(x, type_=TIMESTAMP(timezone=True)),
    "time": lambda x: cast(x, type_=INTERVAL),
    "duration": lambda x: cast(cast(x, TEXT), type_=INTERVAL),
    "point": lambda x: ST_GeomFromGeoJSON(Point(value=x).to_json()),
    "multipoint": lambda x: ST_GeomFromGeoJSON(MultiPoint(value=x).to_json()),
    "linestring": lambda x: ST_GeomFromGeoJSON(LineString(value=x).to_json()),
    "multilinestring": lambda x: ST_GeomFromGeoJSON(
        MultiLineString(value=x).to_dict()
    ),
    "polygon": lambda x: ST_GeomFromGeoJSON(Polygon(value=x).to_json()),
    "box": lambda x: ST_GeomFromGeoJSON(Box(value=x).to_json()),
    "multipolygon": lambda x: ST_GeomFromGeoJSON(
        MultiPolygon(value=x).to_json()
    ),
    "geojson": lambda x: ST_GeomFromGeoJSON(x),
}


def create_cte(
    opstr: str, symbol: Symbol, value: Value | None = None
) -> tuple[TableTypeAlias, CTE]:
    """
    Creates a CTE from a binary expression.

    Parameters
    ----------
    opstr : str
        The expression operator.
    symbol : Symbol
        The lhs of the expression.
    value : Value, optional
        The rhs of the expression, if it exists.

    Returns
    -------
    tuple[TableTypeAlias, CTE]
        A tuple of a table to join on and the CTE.
    """
    if not isinstance(symbol, Symbol):
        raise ValueError(f"CTE passed a symbol with type '{type(symbol)}'.")
    elif not isinstance(value, Value) and value is not None:
        raise ValueError(f"CTE passed a value with type '{type(value)}'.")
    elif value and symbol.type != value.type:
        symbol_type = (
            map_symbol_attribute_to_type[symbol.attribute]
            if symbol.attribute
            else symbol.type
        )
        if symbol_type != value.type:
            raise TypeError(
                f"Type mismatch between symbol and value. {symbol_type} != {value.type}."
            )

    op = map_opstr_to_operator[opstr]
    table, lhs = map_name_to_table_column[symbol.name]
    rhs = (
        map_value_type_to_type_cast[value.type](value.value) if value else None
    )

    # add keying
    if symbol.key:
        lhs = lhs[symbol.key]

        # add type cast
        if not symbol.attribute:
            lhs = map_type_to_jsonb_type_cast[symbol.type](lhs)
        else:
            lhs = map_attribute_to_jsonb_type_cast[symbol.attribute](lhs)

    # add attribute modifier
    if symbol.attribute:
        modifier = map_attribute_to_type_cast[symbol.attribute][symbol.name]
        lhs = modifier(lhs)

    return (table, select(table.id).where(op(lhs, rhs)).cte())


def _recursive_search_logic_tree(
    func: OneArgFunction | TwoArgFunction | NArgFunction,
    cte_list: list | None = None,
    tables: list[TableTypeAlias] | None = None,
) -> tuple[int | dict, list[CTE], list[TableTypeAlias]]:
    """
    Walks the filtering function to produce dependencies.
    """
    if not isinstance(func, OneArgFunction | TwoArgFunction | NArgFunction):
        raise TypeError(
            f"Expected input to be of type 'OneArgFunction | TwoArgFunction | NArgFunction'. Received '{func}'."
        )
    cte_list = cte_list if cte_list else list()
    tables = tables if tables else list()
    logical_tree = dict()

    if isinstance(func, OneArgFunction):
        if isinstance(func.arg, Symbol):
            table, cte = create_cte(opstr=func.op, symbol=func.arg)
            tables.append(table)
            cte_list.append(cte)
            return (len(cte_list) - 1, cte_list, tables)
        else:
            branch, cte_list, tables = _recursive_search_logic_tree(
                func.arg, cte_list, tables
            )
            logical_tree[func.op] = branch
            return (logical_tree, cte_list, tables)

    elif isinstance(func, TwoArgFunction):
        table, cte = create_cte(opstr=func.op, symbol=func.lhs, value=func.rhs)
        tables.append(table)
        cte_list.append(cte)
        return (len(cte_list) - 1, cte_list, tables)

    elif isinstance(func, NArgFunction):
        branches = list()
        for arg in func.args:
            branch, cte_list, tables = _recursive_search_logic_tree(
                arg, cte_list, tables
            )
            branches.append(branch)
        logical_tree[func.op] = branches
        return (logical_tree, cte_list, tables)


def map_filter_to_tables(
    filter_: Filter | None, label_source: LabelSourceAlias
) -> set[TableTypeAlias]:
    """
    Maps a filter to a set of required tables.

    Parameters
    ----------
    filter_ : Filter
        The filter to search.
    label_source : LabelSourceAlias
        The table to use as a source of labels.

    Returns
    -------
    set[TableTypeAlias]
        The set of tables required by the filter.
    """
    tables = set()
    if filter_ is not None:
        if filter_.datasets:
            tables.add(Dataset)
        if filter_.models:
            tables.add(Model)
        if filter_.datums:
            tables.add(Datum)
        if filter_.annotations:
            tables.add(Annotation)
        if filter_.groundtruths:
            table = GroundTruth if label_source is not Prediction else Datum
            tables.add(table)
        if filter_.predictions:
            table = Prediction if label_source is not GroundTruth else Datum
            tables.add(table)
        if filter_.labels:
            tables.add(Label)
        if filter_.embeddings:
            tables.add(Embedding)
    return tables


def generate_dependencies(
    func: OneArgFunction | TwoArgFunction | NArgFunction | None,
) -> tuple[int | dict | None, list[CTE], list[TableTypeAlias]]:
    """
    Recursively generates the dependencies for creating a filter subquery.

    Parameters
    ----------
    func : OneArgFunction | TwoArgFunction | NArgFunction, optional
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

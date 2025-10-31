from datetime import datetime
from enum import StrEnum

import pyarrow as pa
import pyarrow.lib as pl


class DataType(StrEnum):
    INTEGER = "int"
    FLOAT = "float"
    STRING = "string"
    TIMESTAMP = "timestamp"

    def to_py(self):
        """Get python type."""
        match self:
            case DataType.INTEGER:
                return int
            case DataType.FLOAT:
                return float
            case DataType.STRING:
                return str
            case DataType.TIMESTAMP:
                return datetime

    def to_arrow(self):
        """Get arrow type."""
        match self:
            case DataType.INTEGER:
                return pa.int64()
            case DataType.FLOAT:
                return pa.float64()
            case DataType.STRING:
                return pa.string()
            case DataType.TIMESTAMP:
                return pa.timestamp("us")


def convert_type_mapping_to_fields(
    type_mapping: dict[str, DataType] | None
) -> list[tuple[str, pl.DataType]]:
    """
    Convert type mapping to a pyarrow schema input.

    Parameters
    ----------
    type_mapping : dict[str, DataType] | None
        A map from string key to datatype. Treats input of `None` as empty mapping.

    Returns
    -------
    list[tuple[str, pyarrow.lib.DataType]]
        A list of field name, field type pairs that can be used as input to pyarrow.schema.
    """
    if not type_mapping:
        return []
    return [(k, DataType(v).to_arrow()) for k, v in type_mapping.items()]

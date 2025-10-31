from datetime import datetime

import pyarrow as pa

from valor_lite.cache.datatype import DataType, convert_type_mapping_to_fields


def test_datatype_casting_to_arrow():
    assert DataType.FLOAT.to_arrow() == pa.float64()
    assert DataType.INTEGER.to_arrow() == pa.int64()
    assert DataType.STRING.to_arrow() == pa.string()
    assert DataType.TIMESTAMP.to_arrow() == pa.timestamp("us")


def test_datatype_casting_to_python():
    assert DataType.FLOAT.to_py() is float
    assert DataType.INTEGER.to_py() is int
    assert DataType.STRING.to_py() is str
    assert DataType.TIMESTAMP.to_py() is datetime


def test_convert_type_mapping_to_fields():
    x = convert_type_mapping_to_fields(
        {
            "a": DataType.FLOAT,
            "b": DataType.STRING,
        }
    )
    assert x == [
        ("a", pa.float64()),
        ("b", pa.string()),
    ]

    assert convert_type_mapping_to_fields({}) == []
    assert convert_type_mapping_to_fields(None) == []

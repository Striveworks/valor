import pyarrow as pa
import pytest

from valor_lite.semantic_segmentation.shared import (
    decode_metadata_fields,
    encode_metadata_fields,
    generate_schema,
)


def test_generate_schema_conflicting():
    fields = [
        ("field_a", pa.string()),
        ("field_b", "string"),
        ("count", pa.int64()),
    ]
    with pytest.raises(ValueError) as e:
        generate_schema(fields)
    assert "metadata fields {'count'}" in str(e)


def test_metadata_codec():
    fields = [
        ("field_a", pa.string()),
        ("field_b", "string"),
        ("field_c", pa.int64()),
        ("field_d", pa.float64()),
        ("field_e", pa.bool_()),
        ("field_f", "bool"),
        ("field_g", pa.timestamp("ms")),
        ("field_h", "timestamp[ms]"),
    ]

    encoded_fields = encode_metadata_fields(fields)
    assert encoded_fields == {
        "field_a": "string",
        "field_b": "string",
        "field_c": "int64",
        "field_d": "double",
        "field_e": "bool",
        "field_f": "bool",
        "field_g": "timestamp[ms]",
        "field_h": "timestamp[ms]",
    }

    decoded_spec = decode_metadata_fields(encoded_fields)

    assert decoded_spec == fields

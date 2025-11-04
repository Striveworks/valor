import pyarrow as pa

from valor_lite.semantic_segmentation.shared import Base


def test_metadata_codec():
    spec1 = [
        ("field_a", pa.string()),
        ("field_b", "string"),
        ("field_c", pa.int64()),
    ]
    spec2 = [
        ("field_d", pa.float64()),
        ("field_e", pa.bool_()),
        ("field_f", "bool"),
    ]
    spec3 = [
        ("field_g", pa.timestamp("ms")),
        ("field_h", "timestamp[ms]"),
    ]

    encoded_fields = Base._encode_metadata_fields(
        datum_metadata_fields=spec1,
        groundtruth_metadata_fields=spec2,
        prediction_metadata_fields=spec3,
    )
    assert encoded_fields == {
        "datum": {
            "field_a": "string",
            "field_b": "string",
            "field_c": "int64",
        },
        "groundtruth": {
            "field_d": "double",
            "field_e": "bool",
            "field_f": "bool",
        },
        "prediction": {
            "field_g": "timestamp[ms]",
            "field_h": "timestamp[ms]",
        },
    }

    (
        decoded_spec1,
        decoded_spec2,
        decoded_spec3,
    ) = Base._decode_metadata_fields(encoded_fields)

    assert decoded_spec1 == spec1
    assert decoded_spec2 == spec2
    assert decoded_spec3 == spec3


def test_delete_at_path_edge_case():
    Base.delete_at_path("./path/that/does/not/exist")

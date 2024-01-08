""" These tests make sure that the enums in the client and
in the backend repo remain the same. Though maybe instead of duplicating
the backend should depend on the client?
"""

from enum import EnumMeta

from velour import enums
from velour_api import enums as backend_enums


def _enum_to_dict(enum: EnumMeta) -> dict:
    return {x.name: x.value for x in enum}


def test_tasktype_enum():
    assert _enum_to_dict(enums.TaskType) == _enum_to_dict(
        backend_enums.TaskType
    )


def test_datatypes_enum():
    assert _enum_to_dict(enums.DataType) == _enum_to_dict(
        backend_enums.DataType
    )


def test_annotation_types_enum():
    assert _enum_to_dict(enums.AnnotationType) == _enum_to_dict(
        backend_enums.AnnotationType
    )


def test_evaluation_status_enum():
    assert _enum_to_dict(enums.EvaluationStatus) == _enum_to_dict(
        backend_enums.EvaluationStatus
    )


def test_table_status_enum():
    assert _enum_to_dict(enums.TableStatus) == _enum_to_dict(
        backend_enums.TableStatus
    )


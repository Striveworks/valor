""" These tests make sure that the enums in the client and
in the backend repo remain the same. Though maybe instead of duplicating
the backend should depend on the client?
"""

from enum import EnumMeta

from velour import metrics
from velour_api import enums as backend_enums


def _enum_to_dict(enum: EnumMeta) -> dict:
    return {x.name: x.value for x in enum}


def test_task_enum():
    assert _enum_to_dict(metrics.Task) == _enum_to_dict(backend_enums.Task)

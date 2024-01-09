import pytest

from velour_api.enums import (
    AnnotationType,
    EvaluationStatus,
    ModelStatus,
    TableStatus,
)


def test_annotation_type_members():
    # verify that the enum hasnt changed
    assert {e.value for e in AnnotationType} == {
        AnnotationType.NONE,
        AnnotationType.BOX,
        AnnotationType.POLYGON,
        AnnotationType.MULTIPOLYGON,
        AnnotationType.RASTER,
    }

    # test `numeric`
    assert AnnotationType.NONE.numeric == 0
    assert AnnotationType.BOX.numeric == 1
    assert AnnotationType.POLYGON.numeric == 2
    assert AnnotationType.MULTIPOLYGON.numeric == 3
    assert AnnotationType.RASTER.numeric == 4

    # test `__gt__`
    AnnotationType.RASTER > AnnotationType.MULTIPOLYGON
    AnnotationType.RASTER > AnnotationType.POLYGON
    AnnotationType.RASTER > AnnotationType.BOX
    AnnotationType.RASTER > AnnotationType.NONE
    AnnotationType.MULTIPOLYGON > AnnotationType.POLYGON
    AnnotationType.MULTIPOLYGON > AnnotationType.BOX
    AnnotationType.MULTIPOLYGON > AnnotationType.NONE
    AnnotationType.POLYGON > AnnotationType.BOX
    AnnotationType.POLYGON > AnnotationType.NONE
    AnnotationType.BOX > AnnotationType.NONE
    for e in AnnotationType:
        with pytest.raises(TypeError):
            e > 1234

    # test `__lt__`
    AnnotationType.NONE < AnnotationType.RASTER
    AnnotationType.NONE < AnnotationType.MULTIPOLYGON
    AnnotationType.NONE < AnnotationType.POLYGON
    AnnotationType.NONE < AnnotationType.BOX
    AnnotationType.BOX < AnnotationType.RASTER
    AnnotationType.BOX < AnnotationType.MULTIPOLYGON
    AnnotationType.BOX < AnnotationType.POLYGON
    AnnotationType.POLYGON < AnnotationType.RASTER
    AnnotationType.POLYGON < AnnotationType.MULTIPOLYGON
    AnnotationType.MULTIPOLYGON < AnnotationType.RASTER
    for e in AnnotationType:
        with pytest.raises(TypeError):
            e < 1234

    # test `__ge__`
    AnnotationType.RASTER >= AnnotationType.RASTER
    AnnotationType.RASTER >= AnnotationType.MULTIPOLYGON
    AnnotationType.RASTER >= AnnotationType.POLYGON
    AnnotationType.RASTER >= AnnotationType.BOX
    AnnotationType.RASTER >= AnnotationType.NONE
    AnnotationType.MULTIPOLYGON >= AnnotationType.MULTIPOLYGON
    AnnotationType.MULTIPOLYGON >= AnnotationType.POLYGON
    AnnotationType.MULTIPOLYGON >= AnnotationType.BOX
    AnnotationType.MULTIPOLYGON >= AnnotationType.NONE
    AnnotationType.POLYGON >= AnnotationType.POLYGON
    AnnotationType.POLYGON >= AnnotationType.BOX
    AnnotationType.POLYGON >= AnnotationType.NONE
    AnnotationType.BOX >= AnnotationType.BOX
    AnnotationType.BOX >= AnnotationType.NONE
    AnnotationType.NONE >= AnnotationType.NONE
    for e in AnnotationType:
        with pytest.raises(TypeError):
            e >= 1234

    # test `__le__`
    AnnotationType.NONE <= AnnotationType.RASTER
    AnnotationType.NONE <= AnnotationType.MULTIPOLYGON
    AnnotationType.NONE <= AnnotationType.POLYGON
    AnnotationType.NONE <= AnnotationType.BOX
    AnnotationType.NONE <= AnnotationType.NONE
    AnnotationType.BOX <= AnnotationType.RASTER
    AnnotationType.BOX <= AnnotationType.MULTIPOLYGON
    AnnotationType.BOX <= AnnotationType.POLYGON
    AnnotationType.BOX <= AnnotationType.BOX
    AnnotationType.POLYGON <= AnnotationType.RASTER
    AnnotationType.POLYGON <= AnnotationType.MULTIPOLYGON
    AnnotationType.POLYGON <= AnnotationType.POLYGON
    AnnotationType.MULTIPOLYGON <= AnnotationType.RASTER
    AnnotationType.MULTIPOLYGON <= AnnotationType.MULTIPOLYGON
    AnnotationType.RASTER <= AnnotationType.RASTER
    for e in AnnotationType:
        with pytest.raises(TypeError):
            e <= 1234


def test_table_status_members():
    # verify that the enum hasnt changed
    assert {e.value for e in TableStatus} == {
        TableStatus.CREATING,
        TableStatus.FINALIZED,
        TableStatus.DELETING,
    }

    # test `next`
    assert TableStatus.CREATING.next() == {
        TableStatus.CREATING,
        TableStatus.FINALIZED,
        TableStatus.DELETING,
    }
    assert TableStatus.FINALIZED.next() == {
        TableStatus.FINALIZED,
        TableStatus.DELETING,
    }
    assert TableStatus.DELETING.next() == {TableStatus.DELETING}


def test_model_status_members():
    # verify that the enum hasnt changed
    assert {e.value for e in ModelStatus} == {
        ModelStatus.READY,
        ModelStatus.DELETING,
    }

    # test `next`
    assert ModelStatus.READY.next() == {
        ModelStatus.READY,
        ModelStatus.DELETING,
    }
    assert ModelStatus.DELETING.next() == {ModelStatus.DELETING}


def test_evaluation_status_members():
    # verify that the enum hasnt changed
    assert {e.value for e in EvaluationStatus} == {
        EvaluationStatus.PENDING,
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
        EvaluationStatus.FAILED,
        EvaluationStatus.DELETING,
    }

    # test `next`
    assert EvaluationStatus.PENDING.next() == {
        EvaluationStatus.PENDING,
        EvaluationStatus.RUNNING,
    }
    assert EvaluationStatus.RUNNING.next() == {
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
        EvaluationStatus.FAILED,
    }
    assert EvaluationStatus.DONE.next() == {
        EvaluationStatus.DONE,
        EvaluationStatus.DELETING,
    }
    assert EvaluationStatus.FAILED.next() == {
        EvaluationStatus.FAILED,
        EvaluationStatus.RUNNING,
        EvaluationStatus.DELETING,
    }
    assert EvaluationStatus.DELETING.next() == {EvaluationStatus.DELETING}

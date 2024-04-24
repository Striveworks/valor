import pytest

from valor_api.enums import (
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
        AnnotationType.RANKING,
    }

    # test `numeric`
    assert AnnotationType.NONE.numeric == 0
    assert AnnotationType.BOX.numeric == 1
    assert AnnotationType.POLYGON.numeric == 2
    assert AnnotationType.MULTIPOLYGON.numeric == 3
    assert AnnotationType.RASTER.numeric == 4
    assert AnnotationType.RANKING.numeric == 5

    # test `__gt__`
    _ = AnnotationType.RASTER > AnnotationType.MULTIPOLYGON
    _ = AnnotationType.RASTER > AnnotationType.POLYGON
    _ = AnnotationType.RASTER > AnnotationType.BOX
    _ = AnnotationType.RASTER > AnnotationType.NONE
    _ = AnnotationType.MULTIPOLYGON > AnnotationType.POLYGON
    _ = AnnotationType.MULTIPOLYGON > AnnotationType.BOX
    _ = AnnotationType.MULTIPOLYGON > AnnotationType.NONE
    _ = AnnotationType.POLYGON > AnnotationType.BOX
    _ = AnnotationType.POLYGON > AnnotationType.NONE
    _ = AnnotationType.BOX > AnnotationType.NONE
    for e in AnnotationType:
        with pytest.raises(TypeError):
            _ = e > 1234

    # test `__lt__`
    _ = AnnotationType.NONE < AnnotationType.RASTER
    _ = AnnotationType.NONE < AnnotationType.MULTIPOLYGON
    _ = AnnotationType.NONE < AnnotationType.POLYGON
    _ = AnnotationType.NONE < AnnotationType.BOX
    _ = AnnotationType.BOX < AnnotationType.RASTER
    _ = AnnotationType.BOX < AnnotationType.MULTIPOLYGON
    _ = AnnotationType.BOX < AnnotationType.POLYGON
    _ = AnnotationType.POLYGON < AnnotationType.RASTER
    _ = AnnotationType.POLYGON < AnnotationType.MULTIPOLYGON
    _ = AnnotationType.MULTIPOLYGON < AnnotationType.RASTER
    for e in AnnotationType:
        with pytest.raises(TypeError):
            _ = e < 1234

    # test `__ge__`
    _ = AnnotationType.RASTER >= AnnotationType.RASTER
    _ = AnnotationType.RASTER >= AnnotationType.MULTIPOLYGON
    _ = AnnotationType.RASTER >= AnnotationType.POLYGON
    _ = AnnotationType.RASTER >= AnnotationType.BOX
    _ = AnnotationType.RASTER >= AnnotationType.NONE
    _ = AnnotationType.MULTIPOLYGON >= AnnotationType.MULTIPOLYGON
    _ = AnnotationType.MULTIPOLYGON >= AnnotationType.POLYGON
    _ = AnnotationType.MULTIPOLYGON >= AnnotationType.BOX
    _ = AnnotationType.MULTIPOLYGON >= AnnotationType.NONE
    _ = AnnotationType.POLYGON >= AnnotationType.POLYGON
    _ = AnnotationType.POLYGON >= AnnotationType.BOX
    _ = AnnotationType.POLYGON >= AnnotationType.NONE
    _ = AnnotationType.BOX >= AnnotationType.BOX
    _ = AnnotationType.BOX >= AnnotationType.NONE
    _ = AnnotationType.NONE >= AnnotationType.NONE
    for e in AnnotationType:
        with pytest.raises(TypeError):
            _ = e >= 1234

    # test `__le__`
    _ = AnnotationType.NONE <= AnnotationType.RASTER
    _ = AnnotationType.NONE <= AnnotationType.MULTIPOLYGON
    _ = AnnotationType.NONE <= AnnotationType.POLYGON
    _ = AnnotationType.NONE <= AnnotationType.BOX
    _ = AnnotationType.NONE <= AnnotationType.NONE
    _ = AnnotationType.BOX <= AnnotationType.RASTER
    _ = AnnotationType.BOX <= AnnotationType.MULTIPOLYGON
    _ = AnnotationType.BOX <= AnnotationType.POLYGON
    _ = AnnotationType.BOX <= AnnotationType.BOX
    _ = AnnotationType.POLYGON <= AnnotationType.RASTER
    _ = AnnotationType.POLYGON <= AnnotationType.MULTIPOLYGON
    _ = AnnotationType.POLYGON <= AnnotationType.POLYGON
    _ = AnnotationType.MULTIPOLYGON <= AnnotationType.RASTER
    _ = AnnotationType.MULTIPOLYGON <= AnnotationType.MULTIPOLYGON
    _ = AnnotationType.RASTER <= AnnotationType.RASTER
    for e in AnnotationType:
        with pytest.raises(TypeError):
            _ = e <= 1234


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
        EvaluationStatus.FAILED,
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

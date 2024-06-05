import pytest
from sqlalchemy import alias, or_, select

from valor_api.backend import models
from valor_api.backend.query.solvers import (
    _join_annotation_to_label,
    _join_datum_to_groundtruth,
    _join_datum_to_prediction,
    _join_groundtruth_to_datum,
    _join_label_to_annotation,
    _join_prediction_to_datum,
    _solve_graph,
    generate_filter_subquery,
    generate_query,
)


def test__join_label_to_annotation():
    stmt = str(_join_label_to_annotation(select(models.Annotation.id)))
    groundtruth = alias(models.GroundTruth)
    prediction = alias(models.Prediction)
    assert stmt == str(
        select(models.Annotation.id)
        .join(
            groundtruth,
            groundtruth.c.annotation_id == models.Annotation.id,
            isouter=True,
        )
        .join(
            prediction,
            prediction.c.annotation_id == models.Annotation.id,
            isouter=True,
        )
        .join(
            models.Label,
            or_(
                models.Label.id == groundtruth.c.label_id,
                models.Label.id == prediction.c.label_id,
            ),
        )
    )


def test__join_annotation_to_label():
    stmt = str(_join_annotation_to_label(select(models.Label.id)))

    groundtruth = alias(models.GroundTruth)
    prediction = alias(models.Prediction)
    assert stmt == str(
        select(models.Label.id)
        .join(
            groundtruth,
            groundtruth.c.label_id == models.Label.id,
            isouter=True,
        )
        .join(
            prediction, prediction.c.label_id == models.Label.id, isouter=True
        )
        .join(
            models.Annotation,
            or_(
                models.Annotation.id == groundtruth.c.annotation_id,
                models.Annotation.id == prediction.c.annotation_id,
            ),
        )
    )


def test__join_prediction_to_datum():
    stmt = str(_join_prediction_to_datum(select(models.Datum.id)))
    annotation = alias(models.Annotation)
    assert stmt == str(
        select(models.Datum.id)
        .join(
            annotation, annotation.c.datum_id == models.Datum.id, isouter=True
        )
        .join(
            models.Prediction,
            models.Prediction.annotation_id == annotation.c.id,
        )
    )


def test__join_datum_to_prediction():
    stmt = str(_join_datum_to_prediction(select(models.Prediction.id)))
    annotation = alias(models.Annotation)
    assert stmt == str(
        select(models.Prediction.id)
        .join(
            annotation,
            annotation.c.id == models.Prediction.annotation_id,
            isouter=True,
        )
        .join(models.Datum, models.Datum.id == annotation.c.datum_id)
    )


def test__join_groundtruth_to_datum():
    stmt = str(_join_groundtruth_to_datum(select(models.Datum.id)))
    annotation = alias(models.Annotation)
    assert stmt == str(
        select(models.Datum.id)
        .join(
            annotation, annotation.c.datum_id == models.Datum.id, isouter=True
        )
        .join(
            models.GroundTruth,
            models.GroundTruth.annotation_id == annotation.c.id,
        )
    )


def test__join_datum_to_groundtruth():
    stmt = str(_join_datum_to_groundtruth(select(models.GroundTruth.id)))
    annotation = alias(models.Annotation)
    assert stmt == str(
        select(models.GroundTruth.id)
        .join(
            annotation,
            annotation.c.id == models.GroundTruth.annotation_id,
            isouter=True,
        )
        .join(models.Datum, models.Datum.id == annotation.c.datum_id)
    )


def test__solve_graph_validation():
    with pytest.raises(ValueError):
        _solve_graph(
            select_from=models.Annotation,
            label_source=models.Dataset,
            tables=set(),
        )

    # test skip if target is selected table
    assert (
        _solve_graph(
            select_from=models.Dataset,
            label_source=models.Annotation,
            tables={models.Dataset},
        )
        == []
    )

    # create one join
    assert (
        len(
            _solve_graph(
                select_from=models.Dataset,
                label_source=models.Annotation,
                tables={models.Datum},
            )
        )
        == 1
    )


def test_generate_query_validation():
    # test label source validation
    with pytest.raises(ValueError):
        generate_query(
            select_statement=select(models.Label.id),
            args=(models.Label.id,),
            select_from=models.Label,
            label_source=models.Dataset,
        )


def test_generate_filter_subquery_validation():
    # test label source validation
    with pytest.raises(ValueError):
        generate_filter_subquery(
            conditions=None,  # type: ignore - testing
            select_from=models.Annotation,
            label_source=models.Dataset,
            prefix="cte",
        )

    # test that a valid logic tree has been created
    with pytest.raises(ValueError):
        generate_filter_subquery(
            conditions=None,  # type: ignore - testing
            select_from=models.Annotation,
            label_source=models.Annotation,
            prefix="cte",
        )

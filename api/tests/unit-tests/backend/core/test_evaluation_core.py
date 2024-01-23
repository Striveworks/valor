from sqlalchemy import or_

from velour_api.backend import models
from velour_api.backend.core.evaluation import (
    _create_bulk_expression,
    _create_dataset_expr_from_list,
    _create_eval_expr_from_list,
    _create_model_expr_from_list,
)


def test__create_dataset_expr_from_list():
    # test list with single element
    names = ["1"]
    expr = _create_dataset_expr_from_list(names)
    assert str(expr) == str(
        models.Evaluation.datum_filter["dataset_names"].op("?")("1")
    )

    # test list with multiple elements
    names = ["1", "2", "3"]
    expr = _create_dataset_expr_from_list(names)
    assert str(expr) == str(
        or_(
            models.Evaluation.datum_filter["dataset_names"].op("?")("1"),
            models.Evaluation.datum_filter["dataset_names"].op("?")("2"),
            models.Evaluation.datum_filter["dataset_names"].op("?")("3"),
        )
    )

    # test empty list
    assert _create_dataset_expr_from_list([]) is None


def test__create_model_expr_from_list():
    # test list with single element
    names = ["1"]
    expr = _create_model_expr_from_list(names)
    assert str(expr) == str(models.Evaluation.model_name == "1")

    # test list with multiple elements
    names = ["1", "2", "3"]
    expr = _create_model_expr_from_list(names)
    assert str(expr) == str(
        or_(
            models.Evaluation.model_name == "1",
            models.Evaluation.model_name == "2",
            models.Evaluation.model_name == "3",
        )
    )

    # test empty list
    assert _create_model_expr_from_list([]) is None


def test__create_eval_expr_from_list():
    # test list with single element
    ids = [1]
    expr = _create_eval_expr_from_list(ids)
    assert str(expr) == str(models.Evaluation.id == 1)

    # test list with multiple elements
    ids = [1, 2, 3]
    expr = _create_eval_expr_from_list(ids)
    assert str(expr) == str(
        or_(
            models.Evaluation.id == 1,
            models.Evaluation.id == 2,
            models.Evaluation.id == 3,
        )
    )

    # test empty list
    assert _create_eval_expr_from_list([]) is None


def test__create_bulk_expression():
    # test no input
    assert _create_bulk_expression() == []
    assert _create_bulk_expression(None, None, None) == []

    # test dataset expr with single element
    names = ["1"]
    expr = _create_bulk_expression(dataset_names=names)
    assert len(expr) == 1
    assert str(expr[0]) == str(
        models.Evaluation.datum_filter["dataset_names"].op("?")("1")
    )

    # test dataset expr with multiple elements
    names = ["1", "2", "3"]
    expr = _create_bulk_expression(dataset_names=names)
    assert len(expr) == 1
    assert str(expr[0]) == str(
        or_(
            models.Evaluation.datum_filter["dataset_names"].op("?")("1"),
            models.Evaluation.datum_filter["dataset_names"].op("?")("2"),
            models.Evaluation.datum_filter["dataset_names"].op("?")("3"),
        )
    )

    # test model expr with single element
    names = ["1"]
    expr = _create_bulk_expression(model_names=names)
    assert len(expr) == 1
    assert str(expr[0]) == str(models.Evaluation.model_name == "1")

    # test model expr with multiple elements
    names = ["1", "2", "3"]
    expr = _create_bulk_expression(model_names=names)
    assert len(expr) == 1
    assert str(expr[0]) == str(
        or_(
            models.Evaluation.model_name == "1",
            models.Evaluation.model_name == "2",
            models.Evaluation.model_name == "3",
        )
    )

    # test eval expr with single element
    ids = [1]
    expr = _create_bulk_expression(ids)
    assert len(expr) == 1
    assert str(expr[0]) == str(models.Evaluation.id == 1)

    # test eval expr with multiple elements
    ids = [1, 2, 3]
    expr = _create_bulk_expression(ids)
    assert len(expr) == 1
    assert str(expr[0]) == str(
        or_(
            models.Evaluation.id == 1,
            models.Evaluation.id == 2,
            models.Evaluation.id == 3,
        )
    )

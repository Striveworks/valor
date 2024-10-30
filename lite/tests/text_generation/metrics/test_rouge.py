import pytest
from valor_lite.text_generation import Evaluator, QueryResponse
from valor_lite.text_generation.computation import calculate_rouge_scores


def test_rouge_no_context(mock_client):

    evaluator = Evaluator(client=mock_client)
    with pytest.raises(ValueError):
        assert evaluator.compute_rouge(
            response=QueryResponse(
                query="a",
                response="a",
            )
        )


def test_calculate_rouge_scores():

    rouge_types = [
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
    ]

    # perfect match
    assert calculate_rouge_scores(
        prediction="Mary loves Joe",
        references=[
            "Mary loves Joe",
        ],
        rouge_types=rouge_types,
        use_stemmer=False,
    ) == {
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
        "rougeLsum": 1.0,
    }

    # perfect match, case sensitive
    assert calculate_rouge_scores(
        prediction="MARY LOVES JOE",
        references=[
            "Mary loves Joe",
        ],
        rouge_types=rouge_types,
        use_stemmer=False,
    ) == {
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
        "rougeLsum": 1.0,
    }

    # perfect match, case sensitive
    assert calculate_rouge_scores(
        prediction="Mary loves Joe",
        references=[
            "MARY LOVES JOE",
        ],
        rouge_types=rouge_types,
        use_stemmer=False,
    ) == {
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
        "rougeLsum": 1.0,
    }

    # off by one
    assert calculate_rouge_scores(
        prediction="Mary loves Joe",
        references=["Mary loves Jane"],
        rouge_types=rouge_types,
        use_stemmer=False,
    ) == {
        "rouge1": 2 / 3,
        "rouge2": 0.5,
        "rougeL": 2 / 3,
        "rougeLsum": 2 / 3,
    }

    # incorrect match without stemming
    assert calculate_rouge_scores(
        prediction="flipping the roaring white dolphin",
        references=["flip the roaring white dolphin"],
        rouge_types=rouge_types,
        use_stemmer=False,
    ) == {
        "rouge1": 0.8000000000000002,
        "rouge2": 0.75,
        "rougeL": 0.8000000000000002,
        "rougeLsum": 0.8000000000000002,
    }

    # correct match with stemming
    assert calculate_rouge_scores(
        prediction="flipping the roaring white dolphin",
        references=["flip the roaring white dolphin"],
        rouge_types=rouge_types,
        use_stemmer=True,
    ) == {
        "rouge1": 1,
        "rouge2": 1,
        "rougeL": 1,
        "rougeLsum": 1,
    }

    # test multiple references
    assert calculate_rouge_scores(
        prediction="flipping the roaring white dolphin",
        references=[
            "some random sentence",
            "some other sentence",
            "some final reference",
            "flip the roaring white dolphin",
        ],
        rouge_types=rouge_types,
        use_stemmer=True,
    ) == {
        "rouge1": 1,
        "rouge2": 1,
        "rougeL": 1,
        "rougeLsum": 1,
    }

    # references isn't a list
    assert calculate_rouge_scores(
        prediction="Mary loves Joe",
        references="Mary loves Joe",
        rouge_types=rouge_types,
    ) == {
        "rouge1": 1,
        "rouge2": 1,
        "rougeL": 1,
        "rougeLsum": 1,
    }

    # predictions as a list
    with pytest.raises(ValueError):
        calculate_rouge_scores(
            prediction=["Mary loves Joe"],  # type: ignore - testing
            references=["Mary loves June"],
            rouge_types=rouge_types,
        )

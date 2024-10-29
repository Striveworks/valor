import pytest
from valor_lite.text_generation.computation import calculate_sentence_bleu


def test_calculate_sentence_bleu():

    # perfect match
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Joe"],
            weights=(1,),
        )
        == 1.0
    )

    # perfect match, weights are a list
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Joe"],
            weights=[1],
        )
        == 1.0
    )

    # perfect match, case sensitive
    assert (
        calculate_sentence_bleu(
            prediction="MARY LOVES JOE",
            references=["Mary loves Joe"],
            weights=(1,),
        )
        == 0.0
    )

    # perfect match, case sensitive
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["MARY LOVES JOE"],
            weights=(1,),
        )
        == 0.0
    )

    # perfect match, case sensitive, BLEU-2
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["MARY LOVES JOE"],
            weights=(0.0, 1.0),
        )
        == 0.0
    )

    # BLEU-2
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Joe"],
            weights=(0, 1),
        )
        == 1.0
    )

    # BLEU-4
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Joe"],
            weights=[0.25] * 4,
        )
        < 1e-9
    )

    # off by one
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Jane"],
            weights=(1,),
        )
        == 2 / 3
    )

    # off by one BLEU-2
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Jane"],
            weights=(0, 1),
        )
        == 0.5
    )

    # off by one BLEU-3
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Jane"],
            weights=(0, 0, 1),
        )
        < 1e-9
    )

    # off by one BLEU-4
    assert (
        calculate_sentence_bleu(
            prediction="Mary loves Joe",
            references=["Mary loves Jane"],
            weights=(0, 0, 0, 1),
        )
        < 1e-9
    )

    # different cases
    assert (
        calculate_sentence_bleu(
            prediction="mary loves joe",
            references=["MARY LOVES JOE"],
            weights=(1,),
        )
        == 0.0
    )

    # different cases BLEU-2
    assert (
        calculate_sentence_bleu(
            prediction="mary loves joe",
            references=["MARY LOVES JOE"],
            weights=[0, 1],
        )
        == 0.0
    )

    # different cases BLEU-10
    assert (
        calculate_sentence_bleu(
            prediction="mary loves joe",
            references=["MARY LOVES JOE"],
            weights=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        )
        == 0.0
    )

    # test multiple references
    assert (
        calculate_sentence_bleu(
            prediction="flip the roaring white dolphin",
            references=[
                "some random sentence",
                "some other sentence",
                "some final reference",
                "flip the roaring white dolphin",
            ],
            weights=[0, 1],
        )
        == 1.0
    )

    # test empty weights
    with pytest.raises(ValueError):
        calculate_sentence_bleu(
            prediction="flip the roaring white dolphin",
            references=[
                "some random sentence",
            ],
            weights=[],
        )

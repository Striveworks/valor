import pytest
from valor_lite.text_generation import Context, Evaluator, QueryResponse

try:
    import mistralai
except ImportError:
    mistralai = None

try:
    import openai
except ImportError:
    openai = None


@pytest.mark.skipif(
    openai is None,
    reason="Openai is not installed.",
)
def test_openai_integration():

    assert Evaluator.openai()

    with pytest.raises(ValueError) as e:
        Evaluator.openai(
            retries=1,
            seed=1,
        )
    assert "Seed is provided, but retries is not 0." in str(e)


@pytest.mark.skipif(
    mistralai is None,
    reason="MistralAI is not installed.",
)
def test_mistral_integration():
    assert Evaluator.mistral()


def test_compute_all(
    mock_client,
    verdicts_two_yes_one_no,
):
    mock_client.returning = verdicts_two_yes_one_no
    evaluator = Evaluator(client=mock_client)
    metrics = evaluator.compute_all(
        response=QueryResponse(
            query="The dog is wagging its tail.",
            response="The dog doesn't like the cat.",
            context=Context(
                groundtruth=[
                    "The dog has never met a cat.",
                    "The dog is happy to see the cat.",
                ],
                prediction=[
                    "The dog has never met a cat.",
                    "The dog is wagging its tail.",
                    "Cats and dogs are common pets.",
                ],
            ),
        )
    )

    actual = {
        mtype: [m.to_dict() for m in mvalues]
        for mtype, mvalues in metrics.items()
    }
    assert actual == {
        "AnswerCorrectness": [
            {
                "type": "AnswerCorrectness",
                "value": 0.8,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "AnswerRelevance": [
            {
                "type": "AnswerRelevance",
                "value": 0.6666666666666666,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "Bias": [
            {
                "type": "Bias",
                "value": 0.6666666666666666,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "ContextPrecision": [
            {
                "type": "ContextPrecision",
                "value": 0.8333333333333333,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "ContextRecall": [
            {
                "type": "ContextRecall",
                "value": 0.6666666666666666,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "ContextRelevance": [
            {
                "type": "ContextRelevance",
                "value": 0.6666666666666666,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "Faithfulness": [
            {
                "type": "Faithfulness",
                "value": 0.6666666666666666,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "Hallucination": [
            {
                "type": "Hallucination",
                "value": 0.6666666666666666,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "SummaryCoherence": [
            {
                "type": "SummaryCoherence",
                "value": 4,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "Toxicity": [
            {
                "type": "Toxicity",
                "value": 0.6666666666666666,
                "parameters": {"evaluator": "mock", "retries": 0},
            }
        ],
        "ROUGE": [
            {
                "type": "ROUGE",
                "value": 0.5333333333333333,
                "parameters": {"rouge_type": "rouge1", "use_stemmer": False},
            },
            {
                "type": "ROUGE",
                "value": 0.30769230769230765,
                "parameters": {"rouge_type": "rouge2", "use_stemmer": False},
            },
            {
                "type": "ROUGE",
                "value": 0.5333333333333333,
                "parameters": {"rouge_type": "rougeL", "use_stemmer": False},
            },
            {
                "type": "ROUGE",
                "value": 0.5333333333333333,
                "parameters": {
                    "rouge_type": "rougeLsum",
                    "use_stemmer": False,
                },
            },
        ],
        "BLEU": [
            {
                "type": "BLEU",
                "value": 0.0,
                "parameters": {"weights": [0.25, 0.25, 0.25, 0.25]},
            }
        ],
    }

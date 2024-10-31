import pytest
from valor_lite.text_generation import ClientWrapper, Metric
from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.manager import llm_guided_metric


class MockEvaluator:
    def __init__(
        self,
        client: ClientWrapper | None = None,
        retries: int = 0,
    ) -> None:
        self.client = client
        self.retries = retries
        self.count = 0

    @llm_guided_metric
    def func1(self):
        return Metric.bias(value=1.0, model_name="mock", retries=self.retries)

    @llm_guided_metric
    def raises_invalid_llm_response_error(self):
        raise InvalidLLMResponseError("abc")

    @llm_guided_metric
    def raises_value_error(self):
        raise ValueError

    @llm_guided_metric
    def succeed_on_third_attempt(self):
        if self.count >= 3:
            return Metric.bias(
                value=1.0, model_name="mock", retries=self.retries
            )
        else:
            self.count += 1
            raise InvalidLLMResponseError("abc")


def test_llm_guided_metric_wrapper(mock_client):

    evaluator = MockEvaluator()

    # test that client of None raises a value error
    evaluator.client = None
    with pytest.raises(ValueError):
        evaluator.func1()

    evaluator.client = mock_client

    assert evaluator.func1().to_dict() == {
        "type": "Bias",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    # test llm response error returns as error metric
    assert evaluator.raises_invalid_llm_response_error().to_dict() == {
        "type": "Error",
        "value": {
            "type": "InvalidLLMResponseError",
            "message": "abc",
        },
        "parameters": {
            "evaluator": "mock",
            "retries": 0,
        },
    }

    # test that other errors get raised normally
    with pytest.raises(ValueError):
        evaluator.raises_value_error()

    # test that lack of model name raises issues (this could happen in a custom wrapper)
    delattr(mock_client, "model_name")
    with pytest.raises(AttributeError):
        evaluator.func1()


def test_llm_guided_metric_retrying(mock_client):

    evaluator = MockEvaluator(client=mock_client, retries=0)

    for i in range(2):
        # test llm response error returns as error metric
        evaluator.retries = i
        assert evaluator.succeed_on_third_attempt().to_dict() == {
            "type": "Error",
            "value": {
                "type": "InvalidLLMResponseError",
                "message": "abc",
            },
            "parameters": {
                "evaluator": "mock",
                "retries": i,
            },
        }

    evaluator.retries = 2
    assert evaluator.succeed_on_third_attempt().to_dict() == {
        "type": "Bias",
        "value": 1.0,
        "parameters": {
            "evaluator": "mock",
            "retries": 2,
        },
    }

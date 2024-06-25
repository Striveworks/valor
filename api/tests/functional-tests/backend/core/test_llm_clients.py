import pytest

from valor_api.backend.core.llm_clients import LLMClient


def test_LLMClient(monkeypatch):
    """Check that this parent class mostly throws NotImplementedErrors, since its methods are intended to be overridden by its children."""

    def _return_valid_response(*args, **kwargs):
        return "5"

    def _return_invalid_response(*args, **kwargs):
        return "some bad response"

    client = LLMClient(api_key=None, model_name="model_name")

    fake_message = [{"key": "value"}]

    with pytest.raises(NotImplementedError):
        client.connect()

    with pytest.raises(NotImplementedError):
        client.process_messages(fake_message)

    with pytest.raises(NotImplementedError):
        client(fake_message)

    # patch __call__ with a valid response
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_valid_response,
    )

    assert 5 == client.coherence("some text")

    # patch __call__ with a valid response
    monkeypatch.setattr(
        "valor_api.backend.core.llm_clients.LLMClient.__call__",
        _return_invalid_response,
    )

    with pytest.raises(ValueError):
        client.coherence("some text")

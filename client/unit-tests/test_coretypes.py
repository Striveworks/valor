from unittest.mock import patch

import pytest

from valor import Client


@patch("time.sleep")
@patch("valor.Client")
def test_timeouts(mock_sleep, mock_client):

    with pytest.raises(TimeoutError):
        Client.delete_dataset(mock_client, name="some_dataset", timeout=1)

    with pytest.raises(TimeoutError):
        Client.delete_model(mock_client, name="some_dataset", timeout=1)

    with pytest.raises(TimeoutError):
        Client.delete_evaluation(mock_client, evaluation_id=1, timeout=1)

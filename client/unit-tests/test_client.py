from unittest.mock import patch

import pytest

from valor.client import (
    _format_request_timeout,
    connect,
    get_connection,
    reset_connection,
)
from valor.exceptions import (
    ClientAlreadyConnectedError,
    ClientConnectionFailed,
    ClientNotConnectedError,
)


def test__format_request_timeout():
    assert _format_request_timeout(timeout=None, default=30) == 30
    assert _format_request_timeout(timeout=60, default=30) == 60
    assert _format_request_timeout(timeout=-1, default=30) is None
    assert _format_request_timeout(timeout=0, default=30) is None
    assert _format_request_timeout(timeout=-0.1, default=30) is None


@patch("valor.client.ClientConnection")
def test_connect(ClientConnection):
    connect(host="host")
    ClientConnection.assert_called_once()

    with pytest.raises(ClientAlreadyConnectedError):
        connect(host="host")

    connect(host="host", reconnect=True)

    ClientConnection.side_effect = ClientConnectionFailed("testing")
    with pytest.raises(ClientConnectionFailed) as e:
        connect(host="host", reconnect=True)
    assert "testing" in str(e)


@patch("valor.client.ClientConnection")
def test_get_connection(ClientConnection):
    reset_connection()

    with pytest.raises(ClientNotConnectedError):
        get_connection()

    connect(host="host")
    ClientConnection.assert_called_once()


@patch("valor.client.ClientConnection")
def test_reset_connection(ClientConnection):
    connect(host="host", reconnect=True)
    assert get_connection() is not None

    reset_connection()

    with pytest.raises(ClientNotConnectedError):
        get_connection()
    connect(host="host")  # test without reconnect arg
    assert get_connection() is not None

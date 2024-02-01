from unittest.mock import patch

import pytest

from velour.client import connect, get_connection, reset_connection
from velour.exceptions import (
    ClientAlreadyConnectedError,
    ClientConnectionFailed,
    ClientNotConnectedError,
)


@patch("velour.client.ClientConnection")
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


@patch("velour.client.ClientConnection")
def test_get_connection(ClientConnection):
    reset_connection()

    with pytest.raises(ClientNotConnectedError):
        get_connection()

    connect(host="host")
    ClientConnection.assert_called_once()


@patch("velour.client.ClientConnection")
def test_reset_connection(ClientConnection):
    connect(host="host", reconnect=True)
    assert get_connection() is not None

    reset_connection()

    with pytest.raises(ClientNotConnectedError):
        get_connection()
    connect(host="host")  # test without reconnect arg
    assert get_connection() is not None

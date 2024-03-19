from unittest.mock import patch

import pytest

from valor.client import _chunk_list, connect, get_connection, reset_connection
from valor.exceptions import (
    ClientAlreadyConnectedError,
    ClientConnectionFailed,
    ClientNotConnectedError,
)


def test__chunk_list():
    # edge case
    data = [{"key": "value"}]
    results = _chunk_list(json_list=data, chunk_size_bytes=10)

    assert results == [data]

    # 100 elements with an average element size of 23.8
    data = [{f"key_{i}": f"value_{i}"} for i in range(100)]

    # standard case
    results = _chunk_list(json_list=data, chunk_size_bytes=1000)
    assert (
        len(results) == 4
    )  # recursively chunked once, which added an extra split
    assert [len(x) for x in results] == [42, 41, 1, 16]

    # edge case with small chunk size
    results = _chunk_list(json_list=data, chunk_size_bytes=1)
    assert results == [[x] for x in data]


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

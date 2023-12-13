""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""

import pytest

from velour.client import Client, ClientException


def test_client():
    bad_url = "localhost:8000"

    with pytest.raises(ValueError):
        Client(host=bad_url)

    bad_url2 = "http://localhost:8111"

    with pytest.raises(Exception):
        Client(host=bad_url2)

    good_url = "http://localhost:8000"

    assert Client(host=good_url)


def test__requests_wrapper(client: Client):
    with pytest.raises(ValueError):
        client._requests_wrapper("get", "/datasets/fake_dataset/status")

    with pytest.raises(AssertionError):
        client._requests_wrapper("bad_method", "datasets/fake_dataset/status")

    with pytest.raises(ClientException):
        client._requests_wrapper("get", "not_an_endpoint")

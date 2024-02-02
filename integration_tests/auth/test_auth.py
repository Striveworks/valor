""" These integration tests should be run with a backend at http://localhost:8000
that has authentication enabled. The following environment variables must
be set for these tests:

VELOUR_USERNAME
VELOUR_PASSWORD
"""

import os

import pytest
import requests

from velour.client import ClientConnection
from velour.exceptions import ClientConnectionFailed


@pytest.fixture
def bearer_token() -> str:
    url = "http://localhost:8000/token"
    data = {
        "username": os.environ["VELOUR_USERNAME"],
        "password": os.environ["VELOUR_PASSWORD"],
    }

    resp = requests.post(url, data=data)

    return resp.json()


def test_auth_client_pos(bearer_token: str):
    """Test that we can successfully authenticate and hit an endpoint"""
    client = ClientConnection(
        host="http://localhost:8000", access_token=bearer_token
    )
    assert isinstance(client.get_datasets(), list)


def test_auth_client_neg_no_token():
    """Test that we get an authentication error when we don't pass
    an access token
    """
    with pytest.raises(ClientConnectionFailed) as exc_info:
        ClientConnection(host="http://localhost:8000")
    assert "Not authenticated" in str(exc_info)


def test_auth_client_neg_invalid_token():
    """Test that we get an unauthorized error when we pass
    an invalid access token
    """
    with pytest.raises(ClientConnectionFailed) as exc_info:
        ClientConnection(host="http://localhost:8000", access_token="asdasd")
    assert "Unauthorized" in str(exc_info)

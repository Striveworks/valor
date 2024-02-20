""" These integration tests should be run with a backend at http://localhost:8000
that has authentication enabled. The following environment variables must
be set for these tests:

VALOR_USERNAME
VALOR_PASSWORD
"""

import os

import pytest
import requests

from valor.client import ClientConnection
from valor.exceptions import ClientConnectionFailed, ClientException


# the environment variables for these fixtures have the suffix
# _FOR_TESTING since the client itself will look for the env variables
# without the suffix
@pytest.fixture
def username() -> str:
    return os.environ["VALOR_USERNAME_FOR_TESTING"]


@pytest.fixture
def password() -> str:
    return os.environ["VALOR_PASSWORD_FOR_TESTING"]


@pytest.fixture
def bearer_token(username: str, password: str) -> str:
    url = "http://localhost:8000/token"
    data = {"username": username, "password": password}

    resp = requests.post(url, data=data)

    return resp.json()


def test_auth_client_bearer_pos(bearer_token: str):
    """Test that we can successfully authenticate via bearer token
    and hit an endpoint"""
    client = ClientConnection(
        host="http://localhost:8000", access_token=bearer_token
    )
    assert isinstance(client.get_datasets(), list)


def test_auth_client_bearer_neg_invalid_token():
    """Test that we get an unauthorized error when we pass
    an invalid access token
    """
    with pytest.raises(ClientConnectionFailed) as exc_info:
        ClientConnection(host="http://localhost:8000", access_token="asdasd")
    assert "Unauthorized" in str(exc_info)


def test_auth_client_creds_pos(username: str, password: str):
    """Test that we can successfully authenticate via bearer token
    and hit an endpoint"""
    client = ClientConnection(
        host="http://localhost:8000", username=username, password=password
    )
    assert isinstance(client.get_datasets(), list)


def test_auth_client_creds_refetch_bearer(username: str, password: str):
    """Test that we can refetch a bearer token"""
    client = ClientConnection(
        host="http://localhost:8000", username=username, password=password
    )
    client.access_token = None
    assert isinstance(client.get_datasets(), list)


def test_auth_client_creds_neg(username: str, password: str):
    """Test that we can successfully authenticate via bearer token
    and hit an endpoint"""
    with pytest.raises(ClientException) as exc_info:
        ClientConnection(
            host="http://localhost:8000",
            username=username,
            password="invalid password",
        )
    assert "Incorrect username or password" in str(exc_info)


def test_auth_client_bearer_neg_no_token_or_creds():
    """Test that we get an authentication error when we don't pass
    an access token
    """
    with pytest.raises(ClientConnectionFailed) as exc_info:
        ClientConnection(host="http://localhost:8000")
    assert "Not authenticated" in str(exc_info)

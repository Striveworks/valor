""" These integration tests should be run with a backend at http://localhost:8000
that has authentication enabled. The following environment variables for auth0 must
be set for these tests:

AUTH0_DOMAIN
AUTH0_AUDIENCE
AUTH0_CLIENT_ID
AUTH0_CLIENT_SECRET
"""

import os

import pytest
import requests

from velour.client import Client, ClientException


@pytest.fixture
def bearer_token() -> str:
    url = f"https://{os.environ['AUTH0_DOMAIN']}/oauth/token"
    data = {
        "client_id": os.environ["AUTH0_CLIENT_ID"],
        "client_secret": os.environ["AUTH0_CLIENT_SECRET"],
        "grant_type": "client_credentials",
        "audience": os.environ["AUTH0_AUDIENCE"],
    }

    resp = requests.post(url, data=data)

    return resp.json()["access_token"]


def test_auth_client_pos(bearer_token: str):
    """Test that we can successfully authenticate and hit an endpoint"""
    client = Client(host="http://localhost:8000", access_token=bearer_token)
    assert isinstance(client.get_datasets(), list)


def test_auth_client_neg_no_token():
    """Test that we get an authentication error when we don't pass
    an access token
    """
    with pytest.raises(ClientException) as exc_info:
        Client(host="http://localhost:8000")
    assert "Not authenticated" in str(exc_info)


def test_auth_client_neg_invalid_token():
    """Test that we get an unauthorized error when we pass
    an invalid access token
    """
    with pytest.raises(ClientException) as exc_info:
        Client(host="http://localhost:8000", access_token="asdasd")
    assert "Unauthorized" in str(exc_info)

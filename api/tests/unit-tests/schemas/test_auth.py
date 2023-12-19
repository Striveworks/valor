import pytest
from pydantic import ValidationError

from velour_api import schemas


def test_auth_User():
    # valid
    schemas.User(email="somestring")
    schemas.User(email="123")
    schemas.User()

    # invalid
    with pytest.raises(ValidationError):
        schemas.User(email=123)


def test_auth_APIVersion():
    # valid
    schemas.APIVersion(api_version="1.1.1")

    # invalid
    with pytest.raises(ValidationError):
        schemas.APIVersion(api_version=1)

    with pytest.raises(ValidationError):
        schemas.APIVersion()

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

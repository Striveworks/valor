from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request

from velour_api import logger
from velour_api.settings import auth_settings


class OptionalHTTPBearer(HTTPBearer):
    """Wraps HTTPBearer to allow no-auth (e.g. for testing).
    See https://github.com/tiangolo/fastapi/discussions/8445.
    """

    async def __call__(self, request: Request):
        if auth_settings.no_auth:
            return None
        ret = await super().__call__(request)
        verify_token(ret)
        return ret


def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticates a user with the given username and password.

    Parameters
    ----------
    username : str
        The username to authenticate.
    password : str
        The password to authenticate.

    Returns
    -------
    bool
        True if the username and password match those in `auth_settings`, False otherwise.
    """
    return (
        username == auth_settings.USERNAME
        and password == auth_settings.PASSWORD
    )


def create_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Creates a JWT from the given data.

    Parameters
    ----------
    data : dict
        The data to encode in the token.
    expires_delta : timedelta, optional
        The amount of time until the token expires if None then defaults to 1 day

    Returns
    -------
    str
        The encoded JWT.
    """
    to_encode = data.copy()

    expires_delta = expires_delta or timedelta(days=1)
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode, auth_settings.SECRET_KEY, algorithm=auth_settings.ALGORITHM
    )
    return encoded_jwt


def verify_token(token: HTTPAuthorizationCredentials | None) -> dict:
    """
    Verifies a JWT and returns the data contained in it.

    Parameters
    ----------
    token : HTTPAuthorizationCredentials
        The bearer token or None. If this is None and we're in a no auth setting, then
        an empty dictionary is returned.

    Returns
    -------
    dict
        The data contained in the token.

    Raises
    ------
    HTTPException
        Raises an HTTPException with status code 401 if there's any error in verifying
        or decoding the token.
    """
    if auth_settings.no_auth:
        if token is not None:
            logger.debug(
                f"`auth_settings.no_auth is true but got a token: {token}"
            )
        return {}

    try:
        payload = jwt.decode(
            token.credentials,
            auth_settings.SECRET_KEY,
            algorithms=[auth_settings.ALGORITHM],
        )
    except Exception as e:
        logger.debug(f"error in `verify_token` with `token={token}`: {e}")
        raise HTTPException(status_code=401)

    return payload

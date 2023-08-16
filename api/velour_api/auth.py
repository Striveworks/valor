import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request

from velour_api import logger
from velour_api.settings import auth_settings

if auth_settings.jwks_url is not None:
    jwks_client = jwt.PyJWKClient(auth_settings.jwks_url)
else:
    jwks_client = None


class OptionalHTTPBearer(HTTPBearer):
    """Wraps HTTPBearer to allow no-auth (e.g. for testing).
    See https://github.com/tiangolo/fastapi/discussions/8445
    """

    async def __call__(self, request: Request):
        if auth_settings.no_auth:
            return None
        ret = await super().__call__(request)
        verify_token(ret)
        return ret


def verify_token(token: HTTPAuthorizationCredentials | None) -> dict:
    """Verifies a token. See https://auth0.com/blog/build-and-secure-fastapi-server-with-auth0/

    Parameters
    ----------
    token
        the bearer token or None. If this is None and we're in a no auth setting, then
        an empty dictionary is returned

    Returns
    -------
    the data contained in the token

    Raises
    ------
    HTTPException
        raies an HTTPException with status code 401 if there's any error in verifying
        or decoding the token
    """
    if auth_settings.no_auth:
        if token is not None:
            logger.debug(
                f"`auth_settings.no_auth is true but got a token: {token}"
            )
        return {}

    def _handle_error(msg):
        logger.debug(f"error in `verify_token` with `token={token}`: {msg}")
        raise HTTPException(status_code=401)

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(
            token.credentials
        ).key
    except (
        jwt.exceptions.PyJWKClientError,
        jwt.exceptions.DecodeError,
    ) as error:
        _handle_error(error.__str__())

    try:
        payload = jwt.decode(
            token.credentials,
            signing_key,
            algorithms=auth_settings.algorithms,
            audience=auth_settings.audience,
            issuer=auth_settings.issuer,
        )
    except Exception as e:
        return _handle_error(str(e))

    return payload

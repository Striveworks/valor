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
        return await super().__call__(request)


def verify_token(token: HTTPAuthorizationCredentials | None):
    if auth_settings.no_auth:
        if token is not None:
            logger.debug(
                f"`auth_settings.no_auth is true but got a token: {token}"
            )
        return {}
    # https://auth0.com/blog/build-and-secure-fastapi-server-with-auth0/

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

import jwt
from fastapi.security import HTTPBearer
from starlette.requests import Request

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


def verify_token(token: str):
    # https://auth0.com/blog/build-and-secure-fastapi-server-with-auth0/
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token).key
    except jwt.exceptions.PyJWKClientError as error:
        return {"status": "error", "msg": error.__str__()}
    except jwt.exceptions.DecodeError as error:
        return {"status": "error", "msg": error.__str__()}

    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=auth_settings.algorithms,
            audience=auth_settings.api_audience,
            issuer=auth_settings.issuer,
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

    return payload

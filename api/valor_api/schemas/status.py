from pydantic import BaseModel


class APIVersion(BaseModel):
    """
    Defines an API version string which is sent back to the user after their authentication is confirmed.

    Attributes
    ----------
    api_version : str
        The API version.
    """

    api_version: str


class Health(BaseModel):
    """
    Info regarding the health of the service.

    Attributes
    ----------
    status : str
        A short string reassuring the caller that things are okay.
    """

    status: str


class Readiness(BaseModel):
    """
    Info regarding the readiness of the service.

    Attributes
    ----------
    status : str
        A short string reassuring the caller that things are okay.
    """

    status: str

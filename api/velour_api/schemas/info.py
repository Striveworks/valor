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

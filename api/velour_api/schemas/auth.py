from pydantic import BaseModel


class User(BaseModel):
    """
    Defines an authorized user.

    Attributes
    ----------
    email : str
        The user's email address.
    """

    email: str | None = None


class APIVersion(BaseModel):
    """
    Defines an API version string which is sent back to the user after their authentication is confirmed.

    Attributes
    ----------
    api_version : str
        The API version.
    """

    api_version: str

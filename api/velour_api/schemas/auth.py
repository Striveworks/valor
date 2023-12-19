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

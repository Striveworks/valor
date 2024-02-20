from pydantic import BaseModel


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

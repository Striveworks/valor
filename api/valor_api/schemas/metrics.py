from pydantic import BaseModel, Field


class Metric(BaseModel):
    """
    A metric response from the API.

    Attributes
    ----------
    type : str
        The type of metric.
    parameters : dict
        The parameters of the metric.
    value : float
        The value of the metric.
    """

    type: str
    value: int | float | dict
    parameters: dict = Field(default_factory=dict)

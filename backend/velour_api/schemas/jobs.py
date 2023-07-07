from pydantic import BaseModel, Field, Extra
from uuid import uuid4

from velour_api import enums

class Job(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()))
    status: enums.JobStatus = enums.JobStatus.PENDING

    class Config:
        extra = Extra.allow
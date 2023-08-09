from pydantic import BaseModel

from velour_api.enums import JobStatus
from velour_api.exceptions import JobDoesNotExistError, JobStateError


class JobStateflow(BaseModel):
    jobs: dict[int, JobStatus]

    def set_job(self, id: int, state: JobStatus):
        # create new evaluation job
        if id not in self.jobs:
            if state != JobStatus.PENDING:
                raise JobDoesNotExistError(id)
        else:
            # check current state
            current = self.jobs[id]
            if state not in current.next():
                raise JobStateError(id, f"{current} =/=> {state}")

        # update job status
        self.jobs[id] = state

    def remove_job(self, id: int):
        if id not in self.jobs:
            raise JobDoesNotExistError(id)
        elif self.jobs[id] not in [
            JobStatus.DONE,
            JobStatus.FAILED,
        ]:
            raise JobStateError(id, "cannot remove an actively running job.")
        del self.jobs[id]

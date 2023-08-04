from pydantic import BaseModel

from velour_api import enums, exceptions
from velour_api.enums import Stateflow
from velour_api.exceptions import StateflowError


class EvaluationJobs(BaseModel):
    evaluations: dict[int, enums.JobStatus]

    def set_job(self, id: int, state: enums.JobStatus):

        # create new evaluation job
        if id not in self.evaluations:
            if state != enums.JobStatus.PENDING:
                raise exceptions.EvaluationJobStateError(id)
            self.evaluations[id] = state

        # get current state
        current = self.evaluations[id]

        # state flow
        if state not in current.next():
            raise exceptions.EvaluationJobStateError(id)

        # update job status
        self.evaluations[id] = state


class ModelStatus(BaseModel):
    status: Stateflow = Stateflow.NONE

    def set_status(self, status: Stateflow):
        if status not in self.status.next():
            raise StateflowError(
                f"invalid transititon from {self.status} to {status}"
            )
        self.status = status


class DatasetStatus(BaseModel):
    status: Stateflow = Stateflow.NONE
    models: dict[str, ModelStatus] | None = None

    @property
    def evaluating(self) -> bool:
        if self.models is not None:
            for model in self.models:
                if self.models[model] == Stateflow.EVALUATE:
                    return True
        return False

    def set_model_status(self, name: str, status: Stateflow):
        if self.status not in [Stateflow.READY, Stateflow.EVALUATE]:
            raise StateflowError(
                f"no model ops allowed on dataset with state `{self.status}`"
            )

        # init models dictionary
        if self.models is None:
            self.models = {}

        # init model
        if name not in self.models:
            self.models[name] = ModelStatus()

        # set model status
        self.models[name].set_status(status)

        # update dataset status
        self.status = (
            Stateflow.EVALUATE if self.evaluating else Stateflow.READY
        )

    def set_status(self, status: Stateflow):
        if self.evaluating and status != Stateflow.EVALUATE:
            raise Stateflow(
                f"cannot transition to {status} as a evaluation is currently running."
            )
        elif status not in self.status.next():
            raise StateflowError(
                f"invalid transititon from {self.status} to {status}"
            )
        self.status = status


class BackendStatus(BaseModel):
    datasets: dict[str, DatasetStatus] | None = None

    def set_dataset_status(self, dataset_name: str, status: Stateflow):
        # init datasets dictionary
        if self.datasets is None:
            self.datasets = {}

        # set status
        self.datasets[dataset_name].set_status(status)

    def set_model_status(
        self, dataset_name: str, model_name: str, status: Stateflow
    ):
        # check if dataset exists
        if self.datasets is None or dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)

        # set model status
        self.datasets[dataset_name].set_model_status(model_name, status)

    def get_dataset_status(self, dataset_name: str):
        if self.datasets is None:
            raise RuntimeError("datasets object is uninitialized")
        elif dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)

        return self.datasets[dataset_name].status

    def get_model_status(
        self, dataset_name: str, model_name: str
    ) -> Stateflow:
        if self.datasets is None:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        elif dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        elif self.datasets[dataset_name].models is None:
            raise exceptions.ModelDoesNotExistError(model_name)
        elif model_name not in self.datasets[dataset_name]:
            raise exceptions.ModelDoesNotExistError(model_name)

        return self.datasets[dataset_name].models[model_name].status

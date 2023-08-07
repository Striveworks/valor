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
                raise exceptions.EvaluationJobDoesNotExistError(id)
        else:
            # check current state
            current = self.evaluations[id]
            if state not in current.next():
                raise exceptions.EvaluationJobStateError(
                    id, f"{current} =/=> {state}"
                )

        # update job status
        self.evaluations[id] = state

    def remove_job(self, id: int):
        if id not in self.evaluations:
            raise exceptions.EvaluationJobDoesNotExistError(id)
        elif self.evaluations[id] not in [
            enums.JobStatus.DONE,
            enums.JobStatus.FAILED,
        ]:
            raise exceptions.EvaluationJobStateError(
                id, "cannot remove an actively running job."
            )
        del self.evaluations[id]


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
                if self.models[model].status == Stateflow.EVALUATE:
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

    def remove_model(self, name: str):
        if self.models is None:
            raise exceptions.ModelDoesNotExistError(name)
        elif name not in self.models:
            raise exceptions.ModelDoesNotExistError(name)
        elif self.models[name].status != Stateflow.DELETE:
            raise StateflowError(
                f"cannot delete model `{name}` with state `{self.models[name].status}`"
            )
        del self.models[name]

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

        # init dataset
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = DatasetStatus()

        # set status
        self.datasets[dataset_name].set_status(status)

    def set_model_status(
        self,
        status: Stateflow,
        model_name: str,
        dataset_name: str | None = None,
    ):
        # check if dataset exists
        if self.datasets is None:
            raise exceptions.StateflowError("backend stateflow uninitialized")

        # set status to specific inference
        if dataset_name:
            if dataset_name not in self.datasets:
                raise exceptions.DatasetDoesNotExistError(dataset_name)
            self.datasets[dataset_name].set_model_status(model_name, status)
        # set status to all inferences related to model_name
        else:
            for dataset in self.datasets:
                if self.datasets[dataset].models is not None:
                    self.datasets[dataset].set_model_status(model_name, status)

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

    def remove_model(self, model_name: str, dataset_name: str | None = None):
        if self.datasets is None:
            raise exceptions.StateflowError("backend stateflow uninitialized")

        # dataset => model
        if dataset_name:
            if dataset_name not in self.datasets:
                raise exceptions.DatasetDoesNotExistError(dataset_name)
            self.datasets[dataset_name].remove_model(model_name)
        # all datasets => model
        else:
            for dataset in self.datasets:
                if self.datasets[dataset].models is not None:
                    self.datasets[dataset].remove_model(model_name)

    def remove_dataset(self, dataset_name: str):
        if self.datasets is None:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        elif dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        elif self.datasets[dataset_name].status != Stateflow.DELETE:
            raise StateflowError(
                f"cannot delete dataset `{dataset_name}` with state `{self.datasets[dataset_name].status}`"
            )
        del self.datasets[dataset_name]

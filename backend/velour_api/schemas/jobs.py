from pydantic import BaseModel, Field

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


class InferenceStatus(BaseModel):
    status: Stateflow = Stateflow.NONE

    def set_status(self, status: Stateflow):
        if status not in self.status.next():
            raise StateflowError(
                f"invalid transititon from {self.status} to {status}"
            )
        self.status = status


class DatasetStatus(BaseModel):
    status: Stateflow = Stateflow.NONE
    models: dict[str, InferenceStatus] = Field(default_factory=dict)

    @property
    def evaluating(self) -> bool:
        for model in self.models:
            if self.models[model].set_status == Stateflow.EVALUATE:
                return True
        return False

    def set_status(self, status: Stateflow):
        if self.evaluating and self.status != Stateflow.EVALUATE:
            raise StateflowError(
                f"cannot transition to {status} as a evaluation is currently running."
            )
        elif status not in self.status.next():
            raise StateflowError(
                f"invalid transititon from {self.status} to {status}"
            )
        self.status = status

    def set_inference_status(self, model_name: str, status: Stateflow):
        if self.status not in [Stateflow.READY, Stateflow.EVALUATE]:
            raise StateflowError(
                f"no model ops allowed on dataset with state `{self.status}`"
            )
        if model_name not in self.models:
            self.models[model_name] = InferenceStatus()
        self.models[model_name].set_status(status)
        self.set_status(
            Stateflow.EVALUATE if self.evaluating else Stateflow.READY
        )

    def remove_inference(self, model_name: str):
        if model_name not in self.models:
            raise exceptions.ModelDoesNotExistError(model_name)
        elif self.models[model_name].status != Stateflow.DELETE:
            raise StateflowError(
                f"cannot delete model `{model_name}` with state `{self.models[model_name].status}`"
            )
        del self.models[model_name]


class BackendStatus(BaseModel):
    datasets: dict[str, DatasetStatus] = Field(default_factory=dict)

    """ DATASET """

    def get_dataset_status(self, dataset_name: str):
        if dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        return self.datasets[dataset_name].status

    def set_dataset_status(self, dataset_name: str, status: Stateflow):
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = DatasetStatus()
        self.datasets[dataset_name].set_status(status)

    def remove_dataset(self, dataset_name: str):
        if dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        elif self.datasets[dataset_name].status != Stateflow.DELETE:
            raise StateflowError(
                f"cannot delete dataset `{dataset_name}` with state `{self.datasets[dataset_name].status}`"
            )
        del self.datasets[dataset_name]

    """ INFERENCE """

    def get_inference_status(
        self, dataset_name: str, model_name: str
    ) -> Stateflow:
        if dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        elif model_name not in self.datasets[dataset_name].models:
            raise exceptions.ModelDoesNotExistError(model_name)
        return self.datasets[dataset_name].models[model_name].status

    def set_inference_status(
        self,
        status: Stateflow,
        model_name: str,
        dataset_name: str,
    ):
        if dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        self.datasets[dataset_name].set_inference_status(model_name, status)

    def remove_inference(
        self, model_name: str, dataset_name: str | None = None
    ):
        if dataset_name:
            if dataset_name not in self.datasets:
                raise exceptions.DatasetDoesNotExistError(dataset_name)
            self.datasets[dataset_name].remove_inference(model_name)

    def remove_inferences(self, model_name: str):
        for dataset in self.datasets:
            self.datasets[dataset].remove_inference(model_name)

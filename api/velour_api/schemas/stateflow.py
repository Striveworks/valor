from pydantic import BaseModel, Field

from velour_api.enums import JobStatus, State
from velour_api.exceptions import (
    DatasetAlreadyExistsError,
    DatasetDoesNotExistError,
    DatasetFinalizedError,
    DatasetIsEmptyError,
    DatasetNotFinalizedError,
    JobDoesNotExistError,
    JobStateError,
    ModelAlreadyExistsError,
    ModelDoesNotExistError,
    ModelFinalizedError,
    ModelInferencesDoNotExist,
    ModelIsEmptyError,
    ModelNotFinalizedError,
    StateflowError,
)


def _state_transition_error(
    *,
    before: State,
    after: State,
    dataset_name: str,
    model_name: str | None = None,
):
    # attempt to duplicate
    if after == State.NONE:
        if not model_name:
            raise DatasetAlreadyExistsError(dataset_name)
        else:
            raise ModelAlreadyExistsError(model_name)

    # attempt to operate over non-existent structure
    elif before == State.NONE and after != State.CREATE:
        if not model_name:
            raise DatasetIsEmptyError(dataset_name)
        else:
            raise ModelIsEmptyError(model_name)

    # attempt to evaluate before finalizing
    if before == State.CREATE and after == State.EVALUATE:
        if not model_name:
            return DatasetNotFinalizedError(dataset_name)
        else:
            return ModelNotFinalizedError(
                dataset_name=dataset_name, model_name=model_name
            )

    # attempt to interact with data marked for deletion
    elif before == State.DELETE:
        if not model_name:
            return StateflowError(
                f"cannot modify dataset {dataset_name} as it is being deleted"
            )
        else:
            return StateflowError(
                f"cannot modify inferences for model {model_name} on dataset {dataset_name} as they are being deleted"
            )

    # attempt to edit after finalizing
    elif before != State.NONE and after == State.CREATE:
        if not model_name:
            return DatasetFinalizedError(dataset_name)
        else:
            return ModelFinalizedError(
                dataset_name=dataset_name, model_name=model_name
            )

    # generic
    else:
        return StateflowError(f"invalid transititon from {before} to {after}")


class InferenceState(BaseModel):
    status: State = State.NONE
    jobs: dict[int, JobStatus] = Field(default_factory=dict)

    def set_job_status(self, job_id: int, status: JobStatus):
        if job_id not in self.jobs:
            if status != JobStatus.PENDING:
                raise JobDoesNotExistError(job_id)
        else:
            if status not in self.jobs[job_id].next():
                raise JobStateError(
                    id,
                    f"invalid transition from `{self.jobs[job_id]}` to `{status}`",
                )
        self.jobs[job_id] = status

    def get_job_status(self, job_id: int) -> JobStatus | None:
        if job_id not in self.jobs:
            return None
        return self.jobs[job_id]

    def remove_job(self, job_id: int):
        if job_id not in self.jobs:
            raise JobDoesNotExistError(job_id)
        elif self.jobs[job_id] not in [
            JobStatus.DONE,
            JobStatus.FAILED,
        ]:
            raise JobStateError(
                job_id, "cannot remove an actively running job."
            )
        del self.jobs[job_id]


class DatasetState(BaseModel):
    status: State = State.NONE
    models: dict[str, InferenceState] = Field(default_factory=dict)

    @property
    def evaluating(self) -> bool:
        for model_name in self.models:
            if self.models[model_name].status == State.EVALUATE:
                return True
        return False

    def set_inference_status(
        self, dataset_name: str, model_name: str, status: State
    ):
        if model_name not in self.models:
            if status != State.CREATE:
                raise ModelInferencesDoNotExist(
                    dataset_name=dataset_name, model_name=model_name
                )
            self.models[model_name] = InferenceState()
        elif status not in self.models[model_name].status.next():
            raise _state_transition_error(
                before=self.models[model_name].status,
                after=status,
                dataset_name=dataset_name,
                model_name=model_name,
            )
        self.models[model_name].status = status

    def get_inference_status(self, dataset_name: str, model_name: str):
        if model_name not in self.models:
            raise ModelInferencesDoNotExist(
                dataset_name=dataset_name, model_name=model_name
            )
        return self.models[model_name].status

    def remove_inference(self, dataset_name: str, model_name: str):
        if model_name not in self.models:
            raise ModelInferencesDoNotExist(
                dataset_name=dataset_name, model_name=model_name
            )
        del self.models[model_name]


class Stateflow(BaseModel):
    datasets: dict[str, DatasetState] = Field(default_factory=dict)

    """ DATASET """

    def set_dataset_status(self, dataset_name: str, status: State):
        if dataset_name not in self.datasets:
            if status not in [State.NONE, State.CREATE]:
                raise DatasetDoesNotExistError(dataset_name)
            self.datasets[dataset_name] = DatasetState()
        elif (
            self.datasets[dataset_name].evaluating and status != State.EVALUATE
        ):
            raise StateflowError(
                f"cannot transition to {status} as an evaluation is currently running."
            )
        elif status not in self.datasets[dataset_name].status.next():
            raise _state_transition_error(
                before=self.datasets[dataset_name].status,
                after=status,
                dataset_name=dataset_name,
            )
        self.datasets[dataset_name].status = status

    def get_dataset_status(self, dataset_name: str) -> State:
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        return self.datasets[dataset_name].status

    def remove_dataset(self, dataset_name: str):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        del self.datasets[dataset_name]

    """ MODEL """

    def set_model_status(self, model_name: str, status: State):
        for dataset_name in self.datasets:
            if model_name in self.datasets[dataset_name].models:
                self.datasets[dataset_name].set_inference_status(
                    dataset_name, model_name, status=status
                )

    def remove_model(self, model_name: str):
        for dataset_name in self.datasets:
            if model_name in self.datasets[dataset_name].models:
                self.datasets[dataset_name].remove_inference(
                    dataset_name, model_name
                )

    """ INFERENCE """

    def get_inference_status(
        self, dataset_name: str, model_name: str
    ) -> State:
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        return self.datasets[dataset_name].get_inference_status(
            dataset_name, model_name
        )

    def set_inference_status(
        self, dataset_name: str, model_name: str, status: State
    ):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)

        if self.datasets[dataset_name].status == State.CREATE:
            if status not in [State.NONE, State.CREATE, State.DELETE]:
                raise DatasetNotFinalizedError(dataset_name)
            self.datasets[dataset_name].set_inference_status(
                dataset_name=dataset_name,
                model_name=model_name,
                status=status,
            )
        elif self.datasets[dataset_name].status in [
            State.READY,
            State.EVALUATE,
        ]:
            self.datasets[dataset_name].set_inference_status(
                dataset_name=dataset_name,
                model_name=model_name,
                status=status,
            )
            self.set_dataset_status(
                dataset_name=dataset_name,
                status=State.EVALUATE
                if self.datasets[dataset_name].evaluating
                else State.READY,
            )
        else:
            raise StateflowError(
                f"dataset `{dataset_name}` does not support model operations in its current state `{self.datasets[dataset_name].status}`"
            )

    def remove_inference(self, dataset_name: str, model_name: str):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        self.datasets[dataset_name].remove_inference(dataset_name, model_name)

    """ JOBS """

    def set_job_status(
        self,
        dataset_name: str,
        model_name: str,
        job_id: int,
        status: JobStatus,
    ):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        elif model_name not in self.datasets[dataset_name].models:
            raise ModelDoesNotExistError(model_name)
        self.datasets[dataset_name].models[model_name].set_job_status(
            job_id, status
        )

    def get_job_status(
        self,
        dataset_name: str,
        model_name: str,
        job_id: int,
    ):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        elif model_name not in self.datasets[dataset_name].models:
            raise ModelDoesNotExistError(model_name)
        return (
            self.datasets[dataset_name]
            .models[model_name]
            .get_job_status(job_id)
        )

    def remove_job(
        self,
        dataset_name: str,
        model_name: str,
        job_id: int,
    ):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        elif model_name not in self.datasets[dataset_name].models:
            raise ModelDoesNotExistError(model_name)
        self.datasets[dataset_name].models[model_name].remove_job(job_id)

    def get_dataset_jobs(
        self,
        dataset_name: str,
    ) -> dict[str, list[int]]:
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        return {
            model_name: [
                job_id
                for job_id in self.datasets[dataset_name]
                .models[model_name]
                .jobs
            ]
            for model_name in self.datasets[dataset_name].models
        }

    def get_model_jobs(
        self,
        model_name: str,
    ) -> dict[str, list[int]]:
        return {
            dataset_name: [
                job_id
                for job_id in self.datasets[dataset_name]
                .models[model_name]
                .jobs
            ]
            for dataset_name in self.datasets
            if model_name in self.datasets[dataset_name].models
        }

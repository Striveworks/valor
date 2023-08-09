from pydantic import BaseModel, Field

from velour_api.enums import Stateflow
from velour_api.exceptions import (
    DatasetDoesNotExistError,
    DatasetFinalizedError,
    DatasetNotFinalizedError,
    ModelDoesNotExistError,
    ModelFinalizedError,
    ModelNotFinalizedError,
    StateflowError,
)


class InferenceStateflow(BaseModel):
    name: str
    status: Stateflow = Stateflow.NONE

    def set_status(self, *, dataset_name: str, status: Stateflow):
        if status not in self.status.next():
            if (
                self.status == Stateflow.CREATE
                and status == Stateflow.EVALUATE
            ):
                raise ModelNotFinalizedError(
                    dataset_name=dataset_name, model_name=self.name
                )
            elif (
                self.status not in [Stateflow.NONE, Stateflow.CREATE]
                and status == Stateflow.CREATE
            ):
                raise ModelFinalizedError(
                    dataset_name=dataset_name, model_name=self.name
                )
            else:
                raise StateflowError(
                    f"invalid transititon from {self.status} to {status}"
                )
        self.status = status


class DatasetStateflow(BaseModel):
    name: str
    status: Stateflow = Stateflow.NONE
    models: dict[str, InferenceStateflow] = Field(default_factory=dict)

    @property
    def evaluating(self) -> bool:
        for model in self.models:
            if self.models[model].status == Stateflow.EVALUATE:
                return True
        return False

    def set_status(self, status: Stateflow):
        if self.evaluating and status != Stateflow.EVALUATE:
            raise StateflowError(
                f"cannot transition to {status} as a evaluation is currently running."
            )
        elif status not in self.status.next():
            if (
                self.status == Stateflow.CREATE
                and status == Stateflow.EVALUATE
            ):
                raise DatasetNotFinalizedError(self.name)
            elif (
                self.status not in [Stateflow.NONE, Stateflow.CREATE]
                and status == Stateflow.CREATE
            ):
                raise DatasetFinalizedError(self.name)
            else:
                raise StateflowError(
                    f"invalid transititon from {self.status} to {status}"
                )
        self.status = status

    def set_inference_status(self, model_name: str, status: Stateflow):
        if self.status not in [Stateflow.READY, Stateflow.EVALUATE]:
            if self.status == Stateflow.CREATE:
                raise DatasetNotFinalizedError(self.name)
            else:
                raise StateflowError(
                    f"invalid state transition to {status} while model inference operations are running"
                )
        if model_name not in self.models:
            self.models[model_name] = InferenceStateflow(name=model_name)
        self.models[model_name].set_status(
            dataset_name=self.name, status=status
        )
        self.set_status(
            Stateflow.EVALUATE if self.evaluating else Stateflow.READY
        )

    def remove_inference(self, model_name: str):
        if model_name not in self.models:
            raise ModelDoesNotExistError(model_name)
        elif self.models[model_name].status != Stateflow.DELETE:
            raise StateflowError(
                f"cannot delete model `{model_name}` with state `{self.models[model_name].status}`"
            )
        del self.models[model_name]


class BackendStateflow(BaseModel):
    datasets: dict[str, DatasetStateflow] = Field(default_factory=dict)

    """ DATASET """

    def get_dataset_status(self, dataset_name: str):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        return self.datasets[dataset_name].status

    def set_dataset_status(self, dataset_name: str, status: Stateflow):
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = DatasetStateflow(name=dataset_name)
        self.datasets[dataset_name].set_status(status)

    def remove_dataset(self, dataset_name: str):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
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
            raise DatasetDoesNotExistError(dataset_name)
        elif model_name not in self.datasets[dataset_name].models:
            raise ModelDoesNotExistError(model_name)
        return self.datasets[dataset_name].models[model_name].status

    def set_inference_status(
        self,
        status: Stateflow,
        model_name: str,
        dataset_name: str,
    ):
        if dataset_name not in self.datasets:
            raise DatasetDoesNotExistError(dataset_name)
        self.datasets[dataset_name].set_inference_status(model_name, status)

    def remove_inference(
        self, model_name: str, dataset_name: str | None = None
    ):
        if dataset_name:
            if dataset_name not in self.datasets:
                raise DatasetDoesNotExistError(dataset_name)
            self.datasets[dataset_name].remove_inference(model_name)

    def remove_inferences(self, model_name: str):
        for dataset in self.datasets:
            self.datasets[dataset].remove_inference(model_name)

from pydantic import BaseModel

from velour_api import enums, exceptions


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


class BackendState(BaseModel):
    status: enums.Stateflow = enums.Stateflow.CREATE
    models: dict[str, enums.Stateflow] | None = None

    def add_model(self, model_name: str):
        if self.status not in [
            enums.Stateflow.READY,
            enums.Stateflow.EVALUATE,
        ]:
            raise exceptions.StateflowError(
                f"Dataset cannot accept new model in its current state: {self.status}"
            )
        if model_name in self.models:
            if self.models[model_name] != enums.Stateflow.CREATE:
                raise exceptions.ModelAlreadyExistsError(model_name)
            else:
                # do nothing
                return
        if self.models is None:
            self.models = {}
        self.models[model_name] = enums.Stateflow.CREATE

    def update_model(self, model_name: str, status: enums.Stateflow):
        if model_name not in self.models:
            raise exceptions.ModelDoesNotExistError(model_name)

        if self.status not in [
            enums.Stateflow.READY,
            enums.Stateflow.EVALUATE,
        ]:
            raise exceptions.StateflowError(f"dataset has state {self.status}")

        if status not in self.models[model_name].next():
            raise exceptions.StateflowError(
                f"model attempted transition from {self.models[model_name]} to {status}"
            )

        # conditional cases wrt status
        if status == enums.Stateflow.CREATE:
            pass
        elif status == enums.Stateflow.EVALUATE:
            self.update(status)  # dataset should switch to evaluate as well
        elif status == enums.Stateflow.READY:
            pass
        elif status == enums.Stateflow.DELETE:
            pass
        else:
            raise exceptions.StateflowError("Unknown state.")

        # update model
        self.models[model_name] = status

    def remove_model(self, model_name: str):
        if model_name not in self.models:
            raise exceptions.ModelDoesNotExistError(model_name)
        if self.models[model_name] != enums.Stateflow.DELETE:
            raise exceptions.StateflowError(
                f"cannot delete {model_name} as it is currently in state {self.models[model_name]}"
            )
        del self.models[model_name]

    def update(self, status: enums.Stateflow) -> str | None:
        if status not in self.status.next():
            raise exceptions.StateflowError(
                f"dataset attempted transition from {self.status} to {status}"
            )

        # conditional cases wrt status
        if status == enums.Stateflow.CREATE:
            self.models = None
        elif status == enums.Stateflow.READY:
            pass
        elif status == enums.Stateflow.EVALUATE:
            pass
        elif status == enums.Stateflow.DELETE:
            for key in self.models:
                self.models[key].status = enums.Stateflow.DELETE
        else:
            raise exceptions.StateflowError("Unknown state.")

        # update dataset
        self.status = status


class BackendStatus(BaseModel):
    datasets: dict[str, BackendState] | None = None

    """ General """

    def readable(
        self, dataset_name: str, model_name: str | None = None
    ) -> bool:
        if dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)

        if self.datasets[dataset_name].status == enums.Stateflow.DELETE:
            return False

        if model_name:
            if model_name not in self.datasets[dataset_name]:
                raise exceptions.ModelDoesNotExistError(model_name)
            return (
                self.datasets[dataset_name].models[model_name]
                != enums.Stateflow.DELETE
            )
        else:
            return True

    """ Dataset manipulation """

    def update_dataset(self, dataset_name: str, status: enums.Stateflow):
        if self.datasets is None:
            self.datasets = dict()

        if dataset_name not in self.datasets:
            if status == enums.Stateflow.CREATE:
                self.datasets[dataset_name] = BackendState()
            else:
                raise exceptions.DatasetDoesNotExistError(dataset_name)
        else:
            self.datasets[dataset_name].update(status)

    def remove_dataset(self, dataset_name: str):
        if self.datasets is None or dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        # check dataset state
        if self.datasets[dataset_name].status != enums.Stateflow.DELETE:
            raise exceptions.StateflowError(
                f"cannot delete dataset {dataset_name} as it has state {self.datasets[dataset_name].status}"
            )
        # check model states
        for model_name in self.datasets[dataset_name].models:
            if (
                self.datasets[dataset_name].models[model_name]
                != enums.Stateflow.DELETE
            ):
                raise exceptions.StateflowError(
                    f"cannot delete dataset {dataset_name} as model {model_name} has state {self.datasets[dataset_name].models[model_name]}"
                )
        # remove dataset
        del self.datasets[dataset_name]

    """ Model manipulation """

    def update_model(
        self, model_name: str, dataset_name: str, status: enums.Stateflow
    ):
        if self.datasets is None or dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)

        if self.datasets[dataset_name].models is None:
            self.datasets[dataset_name].models = dict()

        if model_name not in self.datasets[dataset_name].models:
            if status == enums.Stateflow.CREATE:
                self.datasets[dataset_name].add_model(model_name)
            else:
                raise exceptions.ModelDoesNotExistError(model_name)
        else:
            self.datasets[dataset_name].update_model(model_name, status)

    def remove_model(self, model_name: str, dataset_name: str):
        if self.datasets is None or dataset_name not in self.datasets:
            raise exceptions.DatasetDoesNotExistError(dataset_name)
        self.datasets[dataset_name].remove_model(model_name)

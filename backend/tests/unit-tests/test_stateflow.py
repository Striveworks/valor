from unittest.mock import MagicMock

import pytest

import velour_api
from velour_api.backend.stateflow import _update_backend_state
from velour_api.enums import Stateflow
from velour_api.exceptions import (
    DatasetFinalizedError,
    DatasetNotFinalizedError,
    StateflowError,
)
from velour_api.schemas.stateflow import (
    BackendStateflow,
    DatasetStateflow,
    InferenceStateflow,
)


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    velour_api.backend.jobs.r = None


def test__update_backend_state():

    velour_api.backend.stateflow.get_backend_state = MagicMock()
    velour_api.backend.stateflow.set_backend_state = MagicMock()

    def _generate_status(
        ds_state: Stateflow | None, md_state: Stateflow | None
    ):
        if ds_state is None:
            return BackendStateflow()

        if md_state is None:
            models = {}
        else:
            models = {"md": InferenceStateflow(name="md", status=md_state)}

        return BackendStateflow(
            datasets={
                "ds": DatasetStateflow(
                    name="ds",
                    status=ds_state,
                    models=models,
                )
            }
        )

    def _test_permutation_neg(
        ds_state: Stateflow,
        ds_next: Stateflow | None,
        md_state: Stateflow | None = None,
        md_next: Stateflow | None = None,
        error=StateflowError,
    ):
        velour_api.backend.stateflow.get_backend_state.return_value = (
            _generate_status(
                ds_state=ds_state,
                md_state=md_state,
            )
        )
        if ds_next is not None:
            with pytest.raises(error):
                _update_backend_state(ds_next, dataset_name="ds")
        else:
            with pytest.raises(error):
                _update_backend_state(
                    md_next, dataset_name="ds", model_name="md"
                )

    def _test_permutation_pos(
        ds_state: Stateflow,
        ds_next: Stateflow,
        md_state: Stateflow | None = None,
        md_next: Stateflow | None = None,
    ):
        velour_api.backend.stateflow.get_backend_state.return_value = (
            _generate_status(
                ds_state=ds_state,
                md_state=md_state,
            )
        )
        if ds_next != Stateflow.NONE:
            _update_backend_state(ds_next, dataset_name="ds")
        else:
            _update_backend_state(md_next, dataset_name="ds", model_name="md")

    # dataset - positive cases
    _test_permutation_pos(Stateflow.NONE, Stateflow.CREATE)
    _test_permutation_pos(Stateflow.NONE, Stateflow.DELETE)
    _test_permutation_pos(Stateflow.CREATE, Stateflow.CREATE)
    _test_permutation_pos(Stateflow.CREATE, Stateflow.READY)
    _test_permutation_pos(Stateflow.CREATE, Stateflow.DELETE)
    _test_permutation_pos(Stateflow.READY, Stateflow.READY)
    _test_permutation_pos(Stateflow.READY, Stateflow.EVALUATE)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE)
    _test_permutation_pos(Stateflow.EVALUATE, Stateflow.READY)
    _test_permutation_pos(Stateflow.EVALUATE, Stateflow.EVALUATE)
    _test_permutation_pos(Stateflow.DELETE, Stateflow.DELETE)

    # dataset - negative cases
    _test_permutation_neg(Stateflow.NONE, Stateflow.READY)
    _test_permutation_neg(Stateflow.NONE, Stateflow.EVALUATE)
    _test_permutation_neg(
        Stateflow.CREATE, Stateflow.EVALUATE, error=DatasetNotFinalizedError
    )
    _test_permutation_neg(
        Stateflow.READY, Stateflow.CREATE, error=DatasetFinalizedError
    )
    _test_permutation_neg(
        Stateflow.EVALUATE, Stateflow.CREATE, error=DatasetFinalizedError
    )
    _test_permutation_neg(Stateflow.EVALUATE, Stateflow.DELETE)
    _test_permutation_neg(
        Stateflow.DELETE,
        Stateflow.CREATE,
    )
    _test_permutation_neg(Stateflow.DELETE, Stateflow.READY)
    _test_permutation_neg(Stateflow.DELETE, Stateflow.EVALUATE)

    # model - positive cases
    for state in [Stateflow.READY, Stateflow.EVALUATE]:
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.CREATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.CREATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.READY
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.DELETE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.READY
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.EVALUATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.DELETE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.READY
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.EVALUATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.DELETE
        )

    # model - negative cases
    for state in [Stateflow.READY, Stateflow.EVALUATE]:
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.EVALUATE
        )
    for state in [Stateflow.CREATE, Stateflow.DELETE]:
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.DELETE
        )

    # model & dataset - postive cases
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.CREATE)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.READY)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.DELETE)

    # model & dataset - negative cases
    _test_permutation_neg(
        Stateflow.READY, Stateflow.DELETE, Stateflow.EVALUATE
    )

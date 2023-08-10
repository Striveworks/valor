from unittest.mock import MagicMock

import pytest

import velour_api
from velour_api.backend.stateflow import _update_backend_state
from velour_api.enums import State
from velour_api.exceptions import (
    DatasetFinalizedError,
    DatasetNotFinalizedError,
    StateflowError,
)
from velour_api.schemas.stateflow import (
    DatasetState,
    InferenceState,
    Stateflow,
)


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    velour_api.backend.jobs.r = None


def test__update_backend_state():

    velour_api.backend.stateflow.get_stateflow = MagicMock()
    velour_api.backend.stateflow.set_stateflow = MagicMock()

    def _generate_status(ds_state: State | None, md_state: State | None):
        if ds_state is None:
            return Stateflow()

        if md_state is None:
            models = {}
        else:
            models = {"md": InferenceState(name="md", status=md_state)}

        return Stateflow(
            datasets={
                "ds": DatasetState(
                    name="ds",
                    status=ds_state,
                    models=models,
                )
            }
        )

    def _test_permutation_neg(
        ds_state: State,
        ds_next: State | None,
        md_state: State | None = None,
        md_next: State | None = None,
        error=StateflowError,
    ):
        velour_api.backend.stateflow.get_stateflow.return_value = (
            _generate_status(
                ds_state=ds_state,
                md_state=md_state,
            )
        )
        if ds_next is not None:
            with pytest.raises(error):
                _update_backend_state(status=ds_next, dataset_name="ds")
        else:
            with pytest.raises(error):
                _update_backend_state(
                    status=md_next, dataset_name="ds", model_name="md"
                )

    def _test_permutation_pos(
        ds_state: State,
        ds_next: State,
        md_state: State | None = None,
        md_next: State | None = None,
    ):
        velour_api.backend.stateflow.get_stateflow.return_value = (
            _generate_status(
                ds_state=ds_state,
                md_state=md_state,
            )
        )
        if ds_next != State.NONE:
            _update_backend_state(status=ds_next, dataset_name="ds")
        else:
            _update_backend_state(
                status=md_next, dataset_name="ds", model_name="md"
            )

    # dataset - positive cases
    _test_permutation_pos(State.NONE, State.CREATE)
    _test_permutation_pos(State.NONE, State.DELETE)
    _test_permutation_pos(State.CREATE, State.CREATE)
    _test_permutation_pos(State.CREATE, State.READY)
    _test_permutation_pos(State.CREATE, State.DELETE)
    _test_permutation_pos(State.READY, State.READY)
    _test_permutation_pos(State.READY, State.EVALUATE)
    _test_permutation_pos(State.READY, State.DELETE)
    _test_permutation_pos(State.EVALUATE, State.READY)
    _test_permutation_pos(State.EVALUATE, State.EVALUATE)
    _test_permutation_pos(State.DELETE, State.DELETE)

    # dataset - negative cases
    _test_permutation_neg(State.NONE, State.READY)
    _test_permutation_neg(State.NONE, State.EVALUATE)
    _test_permutation_neg(
        State.CREATE, State.EVALUATE, error=DatasetNotFinalizedError
    )
    _test_permutation_neg(
        State.READY, State.CREATE, error=DatasetFinalizedError
    )
    _test_permutation_neg(
        State.EVALUATE, State.CREATE, error=DatasetFinalizedError
    )
    _test_permutation_neg(State.EVALUATE, State.DELETE)
    _test_permutation_neg(
        State.DELETE,
        State.CREATE,
    )
    _test_permutation_neg(State.DELETE, State.READY)
    _test_permutation_neg(State.DELETE, State.EVALUATE)

    # model - positive cases
    for state in [State.READY, State.EVALUATE]:
        _test_permutation_pos(state, State.NONE, State.NONE, State.CREATE)
        _test_permutation_pos(state, State.NONE, State.CREATE, State.CREATE)
        _test_permutation_pos(state, State.NONE, State.CREATE, State.READY)
        _test_permutation_pos(state, State.NONE, State.CREATE, State.DELETE)
        _test_permutation_pos(state, State.NONE, State.READY, State.READY)
        _test_permutation_pos(state, State.NONE, State.READY, State.EVALUATE)
        _test_permutation_pos(state, State.NONE, State.READY, State.DELETE)
        _test_permutation_pos(state, State.NONE, State.EVALUATE, State.READY)
        _test_permutation_pos(
            state, State.NONE, State.EVALUATE, State.EVALUATE
        )
        _test_permutation_pos(state, State.NONE, State.DELETE, State.DELETE)

    # model - negative cases
    for state in [State.READY, State.EVALUATE]:
        _test_permutation_neg(state, State.NONE, State.NONE, State.READY)
        _test_permutation_neg(state, State.NONE, State.NONE, State.EVALUATE)
        _test_permutation_neg(state, State.NONE, State.NONE, State.DELETE)
        _test_permutation_neg(state, State.NONE, State.CREATE, State.EVALUATE)
        _test_permutation_neg(state, State.NONE, State.READY, State.CREATE)
        _test_permutation_neg(state, State.NONE, State.EVALUATE, State.CREATE)
        _test_permutation_neg(state, State.NONE, State.EVALUATE, State.DELETE)
        _test_permutation_neg(state, State.NONE, State.DELETE, State.CREATE)
        _test_permutation_neg(state, State.NONE, State.DELETE, State.READY)
        _test_permutation_neg(state, State.NONE, State.DELETE, State.EVALUATE)
    for state in [State.CREATE, State.DELETE]:
        _test_permutation_neg(state, State.NONE, State.NONE, State.CREATE)
        _test_permutation_neg(state, State.NONE, State.CREATE, State.CREATE)
        _test_permutation_neg(state, State.NONE, State.CREATE, State.READY)
        _test_permutation_neg(state, State.NONE, State.CREATE, State.DELETE)
        _test_permutation_neg(state, State.NONE, State.READY, State.READY)
        _test_permutation_neg(state, State.NONE, State.READY, State.EVALUATE)
        _test_permutation_neg(state, State.NONE, State.READY, State.DELETE)
        _test_permutation_neg(state, State.NONE, State.EVALUATE, State.READY)
        _test_permutation_neg(
            state, State.NONE, State.EVALUATE, State.EVALUATE
        )
        _test_permutation_neg(state, State.NONE, State.DELETE, State.DELETE)

    # model & dataset - postive cases
    _test_permutation_pos(State.READY, State.DELETE, State.CREATE)
    _test_permutation_pos(State.READY, State.DELETE, State.READY)
    _test_permutation_pos(State.READY, State.DELETE, State.DELETE)

    # model & dataset - negative cases
    _test_permutation_neg(State.READY, State.DELETE, State.EVALUATE)

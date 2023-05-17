import math

import pytest

from velour.client import ModelBase, _generate_chunks


@pytest.fixture
def chunk_size():
    return 100


def test__generate_chunks(chunk_size: int):
    # Empty list (N=0)
    data = []
    assert len(data) == 0
    chunked_data = [
        arr for arr in _generate_chunks("", data, chunk_size=chunk_size)
    ]
    assert len(chunked_data) == 0
    assert data == [d for chunk in chunked_data for d in chunk]

    # Smallest possible list (N=1)
    data = [1]
    assert len(data) == 1
    chunked_data = [
        arr for arr in _generate_chunks("", data, chunk_size=chunk_size)
    ]
    assert len(chunked_data) == 1
    assert len(chunked_data[0]) == 1
    assert data == [d for chunk in chunked_data for d in chunk]

    # Under 1 chunk list (1 < N < chunksize)
    data = [*range(0, chunk_size - 1)]
    assert len(data) < chunk_size
    chunked_data = [
        arr for arr in _generate_chunks("", data, chunk_size=chunk_size)
    ]
    assert len(chunked_data) == 1
    assert len(chunked_data[0]) == len(data)
    assert data == [d for chunk in chunked_data for d in chunk]

    # Exact chunk (N=chunksize)
    data = [*range(0, chunk_size)]
    assert len(data) == chunk_size
    chunked_data = [
        arr for arr in _generate_chunks("", data, chunk_size=chunk_size)
    ]
    assert len(chunked_data) == 1
    assert len(chunked_data[0]) == len(data)
    assert data == [d for chunk in chunked_data for d in chunk]

    # Multiple of chunksize (N = chunksize * M)
    M = 101
    data = [*range(0, chunk_size * M)]
    assert len(data) == M * chunk_size
    chunked_data = [
        arr for arr in _generate_chunks("", data, chunk_size=chunk_size)
    ]
    assert len(chunked_data) == M
    for chunk in chunked_data:
        assert len(chunk) == chunk_size
    assert data == [d for chunk in chunked_data for d in chunk]

    # Multiple of chunksize N with an offset (N = chunksize * M + K)
    M = 101
    K = math.floor(0.5 * chunk_size)
    data = [*range(0, (chunk_size * M) + K)]
    assert len(data) == (M * chunk_size) + K
    chunked_data = [
        arr for arr in _generate_chunks("", data, chunk_size=chunk_size)
    ]
    assert len(chunked_data) == M + 1
    for chunk in chunked_data[:-1]:
        assert len(chunk) == chunk_size
    assert len(chunked_data[-1]) == K
    assert data == [d for chunk in chunked_data for d in chunk]


def test__group_evaluation_settings():
    eval_settings = [
        {
            "model_name": "model",
            "dataset_name": "dset1",
            "model_pred_task_type": "Classification",
            "dataset_gt_task_type": "Classification",
            "id": 1,
        },
        {
            "model_name": "model",
            "dataset_name": "dset3",
            "model_pred_task_type": "Classification",
            "dataset_gt_task_type": "Other Task",
            "id": 2,
        },
        {
            "model_name": "model",
            "dataset_name": "dset2",
            "model_pred_task_type": "Classification",
            "dataset_gt_task_type": "Classification",
            "id": 3,
        },
    ]

    groupings = ModelBase._group_evaluation_settings(eval_settings)

    assert len(groupings) == 2

    assert groupings == [
        {
            "ids": [1, 3],
            "settings": {
                "model_pred_task_type": "Classification",
                "dataset_gt_task_type": "Classification",
            },
            "datasets": ["dset1", "dset2"],
        },
        {
            "ids": [2],
            "settings": {
                "model_pred_task_type": "Classification",
                "dataset_gt_task_type": "Other Task",
            },
            "datasets": ["dset3"],
        },
    ]

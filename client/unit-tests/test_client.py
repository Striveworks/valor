import math

import pytest

from velour.client import Dataset

@pytest.fixture
def chunk_size():
    return 100

def test__generate_chunks(chunk_size: int):

    # Empty list (N=0)
    data = []
    assert len(data) == 0
    chunked_data = [arr for arr in Dataset(None, "none")._generate_chunks(data, chunk_size=chunk_size)]
    assert len(chunked_data) == 0    
    
    # Smallest possible list (N=1)
    data = [1]
    assert len(data) == 1
    chunked_data = [arr for arr in Dataset(None, "none")._generate_chunks(data, chunk_size=chunk_size)]
    assert len(chunked_data) == 1
    assert len(chunked_data[0]) == 1

    # Under 1 chunk list (1 < N < chunksize)
    data = range(0, chunk_size-1)
    assert len(data) < chunk_size
    chunked_data = [arr for arr in Dataset(None, "none")._generate_chunks(data, chunk_size=chunk_size)]
    assert len(chunked_data) == 1
    assert len(chunked_data[0]) == len(data)

    # Exact chunk (N=chunksize)
    data = range(0, chunk_size)
    assert len(data) == chunk_size
    chunked_data = [arr for arr in Dataset(None, "none")._generate_chunks(data, chunk_size=chunk_size)]
    assert len(chunked_data) == 1
    assert len(chunked_data[0]) == len(data)

    # Multiple of chunksize (N = chunksize * M)
    M = 11
    data = range(0, chunk_size * M)
    assert len(data) == M * chunk_size
    chunked_data = [arr for arr in Dataset(None, "none")._generate_chunks(data, chunk_size=chunk_size)]
    assert len(chunked_data) == M

    idx = 0
    for chunk in chunked_data:
        assert len(chunk) == chunk_size
        for i in range(len(chunk)):
            assert data[idx] == chunk[i]
            idx += 1

    assert idx == len(data)

    # Multiple of chunksize N with an offset (N = chunksize * M + K)
    M = 11
    K = math.floor(0.5 * chunk_size)
    data = range(0, (chunk_size * M) + K)
    assert len(data) == (M * chunk_size) + K
    chunked_data = [arr for arr in Dataset(None, "none")._generate_chunks(data, chunk_size=chunk_size)]
    assert len(chunked_data) == M + 1

    idx = 0
    for chunk in chunked_data[:-1]:
        assert len(chunk) == chunk_size
        for i in range(len(chunk)):
            assert data[idx] == chunk[i]
            idx += 1
    assert len(chunked_data[-1]) == K
    for i in range(len(chunked_data[-1])):
        assert data[idx] == chunked_data[-1][i]
        idx += 1
    
    assert idx == len(data)
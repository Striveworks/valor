import pytest

import valor_lite.exceptions as exc


def test_empty_evaluator_exc():
    with pytest.raises(exc.EmptyEvaluatorError) as e:
        raise exc.EmptyEvaluatorError()
    assert "no data" in str(e)


def test_empty_cache_exc():
    with pytest.raises(exc.EmptyCacheError) as e:
        raise exc.EmptyCacheError()
    assert "no data" in str(e)


def test_empty_cache_error_exc():
    with pytest.raises(exc.EmptyCacheError) as e:
        raise exc.EmptyCacheError()
    assert "no data" in str(e)


def test_cache_error_exc():
    with pytest.raises(exc.InternalCacheError) as e:
        raise exc.InternalCacheError("custom message")
    assert "custom message" in str(e)

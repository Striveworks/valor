""" These tests require a redis instance either running unauthenticated
at localhost:6379 or with the following enviornment variables set:
REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_USERNAME, REDIS_DB
"""

import pytest

from velour_api import jobs
from velour_api.enums import JobStatus
from velour_api.exceptions import JobDoesNotExistError
from velour_api.schemas import EvalJob


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """The setup checks that the redis db is empty and
    the teardown flushes it
    """
    jobs.connect_to_redis()
    if len(jobs.r.keys()) != 0:
        raise RuntimeError("redis database is not-empty")
    yield
    jobs.r.flushdb()


def test_add_job():
    job = EvalJob()
    jobs.add_job(job)

    assert jobs.r.get(job.uid) is not None


def test_get_job():
    """test that we can add a job to redis and get it back and test that
    we get an error if a job with a given uid does not exist
    """
    job = EvalJob()
    jobs.add_job(job)

    retrieved_job = jobs.get_job(job.uid)

    assert retrieved_job.dict() == job.dict()

    with pytest.raises(JobDoesNotExistError) as exc_info:
        jobs.get_job("asdasd")

    assert "Job with uid" in str(exc_info)


def test_wrap_metric_computation():
    """Test that job transition status works"""

    def f():
        assert (
            job.status == JobStatus.PROCESSING == jobs.get_job(job.uid).status
        )
        return [1, 2, 3]

    job, wrapped_f = jobs.wrap_metric_computation(f)

    assert job.status == JobStatus.PENDING == jobs.get_job(job.uid).status
    wrapped_f()
    assert job.status == JobStatus.DONE == jobs.get_job(job.uid).status
    assert (
        job.created_metrics_ids
        == [1, 2, 3]
        == jobs.get_job(job.uid).created_metrics_ids
    )

    def g():
        assert (
            job.status == JobStatus.PROCESSING == jobs.get_job(job.uid).status
        )
        raise Exception

    job, wrapped_g = jobs.wrap_metric_computation(g)
    assert job.status == JobStatus.PENDING == jobs.get_job(job.uid).status
    with pytest.raises(Exception):
        wrapped_g()
    assert job.status == JobStatus.FAILED == jobs.get_job(job.uid).status

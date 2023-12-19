""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""

import pytest

from velour.client import Client, ClientException, _validate_version


def test_client():
    bad_url = "localhost:8000"

    with pytest.raises(ValueError):
        Client(host=bad_url)

    bad_url2 = "http://localhost:8111"

    with pytest.raises(Exception):
        Client(host=bad_url2)

    good_url = "http://localhost:8000"

    assert Client(host=good_url)


def test_version_mismatch_warning(caplog):
    # test client being older than api
    _validate_version(client_version="1.1.1", api_version="9.9.9")

    assert all(
        record.levelname == "WARNING" and "older" in record.message
        for record in caplog.records
    )

    caplog.clear()

    # test client being newer than api
    _validate_version(client_version="9.9.9", api_version="1.1.1")

    assert all(
        record.levelname == "WARNING" and "newer" in record.message
        for record in caplog.records
    )

    caplog.clear()

    # test client and API being the same version
    _validate_version(client_version="1.1.1", api_version="1.1.1")

    assert all(
        record.levelname == "DEBUG"
        and "matches client version" in record.message
        for record in caplog.records
    )
    caplog.clear()

    # test missing client or API versions
    _validate_version(client_version=None, api_version="1.1.1")

    assert all(
        record.levelname == "WARNING"
        and "client isn't versioned" in record.message
        for record in caplog.records
    )
    caplog.clear()

    _validate_version(client_version="1.1.1", api_version=None)

    assert all(
        record.levelname == "WARNING"
        and "API didn't return a version" in record.message
        for record in caplog.records
    )

    caplog.clear()

    # test that semantic versioning works correctly
    # client_version > api_version when comparing strings, but
    # client_version < api_version when comparing semantic versions
    _validate_version(client_version="1.12.2", api_version="1.101.12")

    assert all(
        record.levelname == "WARNING" and "older" in record.message
        for record in caplog.records
    )
    caplog.clear()


def test__requests_wrapper(client: Client):
    with pytest.raises(ValueError):
        client._requests_wrapper("get", "/datasets/fake_dataset/status")

    with pytest.raises(AssertionError):
        client._requests_wrapper("bad_method", "datasets/fake_dataset/status")

    with pytest.raises(ClientException):
        client._requests_wrapper("get", "not_an_endpoint")

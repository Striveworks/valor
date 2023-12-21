import requests


def test_health():
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200


def test_ready():
    resp = requests.get("http://localhost:8000/ready")
    assert resp.status_code == 200

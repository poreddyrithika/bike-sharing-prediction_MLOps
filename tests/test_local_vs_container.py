"""Local API vs the running container must agree. Start the container first:

    docker run -p 8000:8000 bike-sharing-api:model-v1
    python -m pytest tests/test_local_vs_container.py -s
"""
import json
from urllib.request import Request, urlopen

import pytest
from starlette.testclient import TestClient

from api.main import app
from tests.test_api import VALID

URL = "http://localhost:8000"


def _container(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(URL + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    with urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def test_local_matches_container():
    try:
        _container("/health")
    except Exception:
        pytest.skip("No container listening on localhost:8000")
    with TestClient(app) as client:
        local = client.post("/predict", json=VALID).json()
    remote = _container("/predict", VALID)
    print("local:", local, "container:", remote)
    assert abs(local["predicted_rentals"] - remote["predicted_rentals"]) < 1e-6

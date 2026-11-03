import statistics
import time

import pytest
from starlette.testclient import TestClient

from api.main import app
from tests.test_api import VALID


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_single_request_latency(client):
    start = time.perf_counter()
    assert client.post("/predict", json=VALID).status_code == 200
    assert (time.perf_counter() - start) * 1000 < 2000


def test_repeated_request_latency(client):
    latencies = []
    for _ in range(30):
        start = time.perf_counter()
        assert client.post("/predict", json=VALID).status_code == 200
        latencies.append((time.perf_counter() - start) * 1000)
    print(f"\nmean={statistics.mean(latencies):.1f}ms p95={sorted(latencies)[int(0.95 * len(latencies))]:.1f}ms")
    assert statistics.mean(latencies) < 500


def test_batch_is_faster_than_looping(client):
    days = [VALID] * 50
    start = time.perf_counter()
    assert client.post("/predict/batch", json={"days": days}).status_code == 200
    batch_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for day in days:
        client.post("/predict", json=day)
    assert batch_ms < (time.perf_counter() - start) * 1000


def test_reliability_100_requests(client):
    ok = sum(client.post("/predict", json=VALID).status_code == 200 for _ in range(100))
    assert ok == 100

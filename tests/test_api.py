import pytest
from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


VALID = {"dteday": "2012-06-15", "season": 2, "yr": 1, "mnth": 6, "holiday": 0, "weekday": 5, "workingday": 1,
         "weathersit": 1, "temp": 0.65, "atemp": 0.62, "hum": 0.55, "windspeed": 0.15}
WEEKEND = {**VALID, "dteday": "2012-12-23", "season": 4, "mnth": 12, "weekday": 0, "workingday": 0,
           "weathersit": 3, "temp": 0.2, "atemp": 0.2, "hum": 0.9, "windspeed": 0.35}


def test_root(client):
    assert client.get("/").status_code == 200


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_metadata(client):
    d = client.get("/metadata").json()
    assert d["service"] == "Bike Sharing Demand API"
    assert d["model_loaded"] is True and d["preprocessor_loaded"] is True
    assert d["model_name"] and d["model_version"]


def test_valid_prediction(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    value = r.json()["predicted_rentals"]
    assert isinstance(value, float) and 0 <= value < 20000


def test_prediction_without_optional_date(client):
    payload = {k: v for k, v in VALID.items() if k != "dteday"}
    assert client.post("/predict", json=payload).status_code == 200


def test_good_weather_beats_bad_weather(client):
    good = client.post("/predict", json=VALID).json()["predicted_rentals"]
    bad = client.post("/predict", json=WEEKEND).json()["predicted_rentals"]
    assert good > bad


@pytest.mark.parametrize("field,value", [
    ("season", 7), ("yr", 5), ("mnth", 13), ("weekday", 8), ("holiday", 2), ("weathersit", 0),
    ("temp", 1.5), ("hum", -0.1), ("windspeed", 3), ("temp", "hot"), ("season", "1"), ("temp", None),
])
def test_invalid_values_rejected(client, field, value):
    r = client.post("/predict", json={**VALID, field: value})
    body = r.json()
    assert r.status_code == 422
    assert body["status"] == "error" and body["message"] == "Request validation failed"
    assert any(d["field"] == field for d in body["details"])


def test_missing_field_rejected(client):
    payload = {k: v for k, v in VALID.items() if k != "temp"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and any(d["field"] == "temp" for d in r.json()["details"])


def test_unknown_field_rejected(client):
    assert client.post("/predict", json={**VALID, "casual": 100}).status_code == 422


def test_leakage_fields_rejected(client):
    assert client.post("/predict", json={**VALID, "registered": 10}).status_code == 422


def test_contradictory_holiday_workingday_rejected(client):
    assert client.post("/predict", json={**VALID, "holiday": 1, "workingday": 1}).status_code == 422


def test_bad_date_rejected(client):
    assert client.post("/predict", json={**VALID, "dteday": "not-a-date"}).status_code == 422


def test_empty_body_lists_many_errors(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422 and len(r.json()["details"]) >= 11


def test_malformed_json(client):
    r = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_batch_prediction(client):
    r = client.post("/predict/batch", json={"days": [VALID, WEEKEND]})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 2 and len(body["predictions"]) == 2


def test_batch_matches_single(client):
    single = client.post("/predict", json=VALID).json()["predicted_rentals"]
    batch = client.post("/predict/batch", json={"days": [VALID]}).json()["predictions"][0]["predicted_rentals"]
    assert single == batch


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"days": []}).status_code == 422


def test_batch_with_one_invalid_row_rejected(client):
    r = client.post("/predict/batch", json={"days": [VALID, {**VALID, "season": 9}]})
    assert r.status_code == 422


def test_unknown_route_uses_standard_error_shape(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["status"] == "error"


def test_never_negative(client):
    extreme = {**WEEKEND, "yr": 0, "temp": 0.0, "atemp": 0.0, "hum": 1.0, "windspeed": 1.0, "weathersit": 4}
    assert client.post("/predict", json=extreme).json()["predicted_rentals"] >= 0

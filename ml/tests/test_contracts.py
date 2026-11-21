import copy
import json
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError
from netwatch_ml.contracts import validate, ROOT
from netwatch_ml.serving.app import app

@pytest.fixture
def window(): return json.loads((ROOT/"contracts/examples/window.valid.json").read_text())
def test_valid(window): assert validate("window",window)==window
@pytest.mark.parametrize("change",[
 lambda w:w.update(schema_version=2),
 lambda w:w.update(window_end_ns=w["window_start_ns"]),
 lambda w:w.update(window_start_ns="9999999999999999999"),
 lambda w:w["features"].update(out_packets=1.5),
 lambda w:w["features"].update(mean_packet_size=float("nan")),
 lambda w:w.update(extra="bad")])
def test_invalid(window,change):
 change(window)
 with pytest.raises(Exception): validate("window",window)
def test_unready():
 c=TestClient(app)
 assert c.get("/health/live").status_code==200
 assert c.get("/health/ready").status_code==503
 assert c.post("/v1/predict",json={}).status_code==503


@pytest.fixture
def operation_window(window):
    window.update(entity_type="destination_service", entity_id="TCP/192.0.2.1:8080",
                  window_start_ns="5000000000", window_end_ns="10000000000",
                  application={"completed_requests": 2, "server_errors": 0, "mean_latency_ms": 1.,
                               "planned_activity": None, "source_ref": "proxy:42",
                               "operations": [{"route_id": "reports.export", "method": "POST",
                                               "completed_requests": 1, "request_body_bytes": None,
                                               "response_body_bytes": 20, "expected_requests": 0}]})
    return window


def test_operation_context_distinguishes_unknown_from_zero(operation_window):
    app = validate("window", operation_window)["application"]
    assert app["source_ref"] == "proxy:42"
    assert app["operations"][0]["request_body_bytes"] is None
    assert app["operations"][0]["expected_requests"] == 0


@pytest.mark.parametrize("change", [
    lambda app: app.update(source_ref="proxy\n"),
    lambda app: app.update(source_ref="https://proxy"),
    lambda app: app["operations"].append(dict(app["operations"][0], completed_requests=0, response_body_bytes=None)),
    lambda app: app["operations"].append(dict(app["operations"][0], route_id="other", completed_requests=2)),
    lambda app: app["operations"][0].update(completed_requests=0),
    lambda app: app["operations"][0].update(route_id="reports\n"),
    lambda app: app["operations"][0].update(response_body_bytes=1.),
    lambda app: app["operations"][0].pop("expected_requests"),
])
def test_invalid_operation_context(operation_window, change):
    change(operation_window["application"])
    with pytest.raises((ValidationError, ValueError)):
        validate("window", operation_window)

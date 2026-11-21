import json

import pytest

from netwatch_ml.contracts import ROOT, validate
from netwatch_ml.models.detector import FEATURES, predict_window
from netwatch_ml.models.service_detector import APP_FEATURES, PROFILE


def test_context_is_exact_and_unknown_cannot_be_read_as_zero():
    row = json.loads((ROOT / "contracts/examples/window.valid.json").read_text())
    row.update(entity_type="destination_service", entity_id="TCP/127.0.0.1:8080",
               window_start_ns="5000000000", window_end_ns="10000000000")
    row["quality"].update(coverage="complete", capture_dropped=0, late_packets=0, state_evictions=0)
    width = len(FEATURES) + len(APP_FEATURES)
    payload = dict(profile=PROFILE, feature_set_version="traffic-v1", contextual=True, method="mlp",
                   binding=dict(sensor_id=row["sensor_id"], entity_id=row["entity_id"]),
                   application_limits=dict(min_requests=3, server_error_fraction=.2, latency_ms=100),
                   features=FEATURES, thresholds={"planned": .9, "unplanned": .9},
                   preprocessing=dict(median=[0.] * width, scale=[1.] * width),
                   layers=[[[[0.]] * (2 * width), [0.]], [[[0.]], [0.]]])
    model = dict(model_id="test", detector=payload)
    assert predict_window(row, model)["unavailable_reason"] == "application_context_missing"
    healthy = dict(completed_requests=10, server_errors=0, mean_latency_ms=1., planned_activity=True)
    row["application"] = healthy
    validate("window", row)
    assert predict_window(row, model)["status"] == "normal"
    row["application"] = dict(healthy, planned_activity=None)
    assert predict_window(row, model)["unavailable_reason"] == "application_context_missing"
    row["application"]["server_errors"] = 5
    assert predict_window(row, model)["status"] == "anomalous"
    row["application"]["server_errors"] = 11
    with pytest.raises(ValueError, match="invalid application context"):
        validate("window", row)


def test_confirmed_pair_calibration_ignores_isolated_spikes_and_respects_epoch():
    from netwatch_ml.training.service_detector import confirmed_pair_thresholds

    def row(epoch, t, label):
        step = 5_000_000_000
        return (dict(capture_epoch=epoch, window_start_ns=str(t * step), window_end_ns=str((t+1) * step)), label)

    rows = [row("a", i, label) for i, label in enumerate(["normal", "normal", "normal", "anomaly", "anomaly"])]
    rows += [row("b", 3, "normal"), row("c", 4, "normal")]
    thresholds, report = confirmed_pair_thresholds(rows, [.2, .97, .1, .65, .66, .99, .99], False, None)
    assert report == {"false_normal_pairs": 0, "positive_runs_with_pair": 1}
    assert .2 <= thresholds["unplanned"] < .65


def test_independent_http_evidence_survives_a_packet_gap():
    row = json.loads((ROOT / "contracts/examples/window.valid.json").read_text())
    row.update(entity_type="destination_service", entity_id="TCP/127.0.0.1:8080",
               window_start_ns="5000000000", window_end_ns="10000000000")
    row["quality"] = dict(coverage="gap", capture_dropped=None, late_packets=0, state_evictions=0)
    row["application"] = dict(completed_requests=10, server_errors=0, mean_latency_ms=200., planned_activity=None)
    validate("window", row)
    payload = dict(profile=PROFILE, feature_set_version="traffic-v1", binding=dict(
        sensor_id=row["sensor_id"], entity_id=row["entity_id"]),
        application_limits=dict(min_requests=3, server_error_fraction=.2, latency_ms=100))
    model = dict(model_id="test", detector=payload)
    old = predict_window(row, model)
    result = predict_window(row, model, independent_evidence=True)
    validate("prediction", result)
    assert result["status"] == old["status"] == "insufficient_data"
    assert "http" not in old
    assert result["http"] == dict(status="anomalous", reason="latency")
    row["sensor_id"] = "unrelated-sensor"
    assert "http" not in predict_window(row, model, independent_evidence=True)

"""Regression checks for event-time calibration and unsupported schedule regimes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from netwatch_ml.contracts import ROOT, validate
from netwatch_ml.models.application_operations import operation_features, operation_measurement_available
from netwatch_ml.models.detector import FEATURES, predict_window
from netwatch_ml.models.service_detector import PROFILE, context_width, feature_rows, validate_config
from netwatch_ml.training import service_detector
from netwatch_ml.training.baselines import sha

sys.path.insert(0, str(ROOT))
from netwatch_ml.training.incident_calibration import (  # noqa: E402
    calibrate_incidents, load_calibration_timeline, supported_inference_input,
)


NS = 1_000_000_000
BINDING = dict(sensor_id="calibration", entity_id="TCP/127.0.0.1:8080")


@pytest.fixture(scope="module")
def projector(tmp_path_factory):
    binary = tmp_path_factory.mktemp("calibration") / "projector"
    subprocess.run(["go", "build", "-o", str(binary), "./benchmarks/projection-driver"], cwd=ROOT, check=True)
    return binary


def window(capture, index, score, *, planned=False, consecutive=True):
    row = json.loads((ROOT / "contracts/examples/window.valid.json").read_text())
    row.update(BINDING, entity_type="destination_service", event_id=f"{capture}-{index}", capture_epoch=capture,
               window_start_ns=str(index * 5 * NS), window_end_ns=str((index + 1) * 5 * NS))
    row["quality"] = dict(coverage="complete" if consecutive else "gap", capture_dropped=0,
                          late_packets=0, state_evictions=0)
    row["features"]["mean_packet_size"] = score
    row["application"] = dict(completed_requests=5, server_errors=0, mean_latency_ms=2., planned_activity=planned)
    return row


def truth(start, end, kind, planned):
    return dict(start=start * 5 * NS, end=end * 5 * NS, label=kind, planned=planned, evidence_channel="network")


def timeline(capture, *, planned=False, positive_planned=False):
    points = [window(capture, i, score, planned=planned) for i, score in
              enumerate((.05, .8, .8, .4, .4, .8, .8, .05, .05))]
    positive = "normal" if planned and not positive_planned else "surge"
    annotation = [truth(a, b, label, planned) for a, b, label in
                  ((0, 1, "normal"), (1, 3, positive), (3, 5, "normal"), (5, 7, positive), (7, 9, "normal"))]
    return dict(id=capture, service=BINDING["entity_id"], windows=points, truth=annotation)


def calibrate(groups, projector):
    return calibrate_incidents(groups, lambda windows: [w["features"]["mean_packet_size"] for w in windows],
                               binding=BINDING, version="traffic-v1", contextual=True, projector=projector)


def test_calibration_penalizes_merging_with_real_event_time_windows(projector):
    groups = [timeline("u-one"), timeline("u-two"), timeline("p-one", planned=True), timeline("p-two", planned=True)]
    levels, result = calibrate(groups, projector)
    assert .4 <= levels["unplanned"] < .8 and levels["planned"] is None
    assert result["unsupported_regimes"] == ["planned"]
    assert result["support"]["planned"]["positive_captures"] == 0
    assert result["metrics"]["merged"] == result["metrics"]["false_alarms"] == 0
    assert result["metrics"]["known_found"] == 4
    assert result["metrics"]["evaluated_fraction"] == .5
    assert result["metrics"]["by_normal_regime"]["planned"]["normal_evaluated_hours"] == 0


def test_planned_cannot_calibrate_without_separate_positive_captures(projector):
    groups = [timeline("u-one"), timeline("u-two"), timeline("p-one", planned=True, positive_planned=True),
              timeline("p-two", planned=True)]
    levels, result = calibrate(groups, projector)
    assert levels["planned"] is None
    assert result["support"]["planned"]["positive_captures"] == 1
    groups[-1] = timeline("p-two", planned=True, positive_planned=True)
    levels, result = calibrate(groups, projector)
    assert .4 <= levels["planned"] < .8
    assert result["unsupported_regimes"] == []
    assert result["metrics"]["known_found"] == 8


def test_event_calibration_refuses_an_artifact_without_any_network_regime():
    groups = [timeline("p-one", planned=True), timeline("p-two", planned=True)]
    with pytest.raises(ValueError, match="no supported network regime"):
        calibrate(groups, "unused-projector")
    with pytest.raises(ValueError, match="invalid incident calibration thresholds"):
        validate_config(dict(contextual=True, calibration_policy="v4", method="isolation_forest", forest={},
                             thresholds=dict(unplanned=None, planned=None), unsupported_regimes=["planned", "unplanned"]), 16)


def test_calibration_requires_ordered_intact_timeline(tmp_path):
    group = timeline("calibration")
    stored = {"source_sha256": "a" * 64, "windows": group["windows"], "truth": group["truth"]}
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(stored))
    source = dict(source_sha256=stored["source_sha256"], timeline=path.name, timeline_sha256=sha(path))
    rows = [(group["windows"][0], "normal"), (group["windows"][1], "anomaly")]
    assert load_calibration_timeline(source, tmp_path, BINDING, "traffic-v1", rows)["id"] == "calibration"
    stored["windows"].pop(4)
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="timeline provenance"):
        load_calibration_timeline(source, tmp_path, BINDING, "traffic-v1", rows)
    source["timeline_sha256"] = sha(path)
    with pytest.raises(ValueError, match="continuous ordered windows"):
        load_calibration_timeline(source, tmp_path, BINDING, "traffic-v1", rows)


@pytest.mark.parametrize("threshold", [1., None])
def test_unsupported_or_legacy_impossible_threshold_preserves_http_signal(threshold):
    row = window("test", 0, .5, planned=True)
    row["application"]["server_errors"] = 5
    bundle = dict(model_id="test", detector=dict(
        profile=PROFILE, feature_set_version="traffic-v1", binding=BINDING, contextual=True,
        thresholds=dict(planned=threshold, unplanned=.5),
        application_limits=dict(min_requests=3, server_error_fraction=.2, latency_ms=100),
    ))
    result = validate("prediction", predict_window(row, bundle, independent_evidence=True))
    assert result["status"] == "insufficient_data" and result["score"] is None
    assert result["unavailable_reason"] == "regime_uncalibrated"
    assert result["http"] == dict(status="anomalous", reason="error_fraction")


def test_event_model_uses_context_values_without_schedule_or_route_identity():
    row = window("test", 0, .5, planned=True)
    app = row["application"]
    app["operations"] = [
        dict(route_id="checkout", method="GET", completed_requests=2, request_body_bytes=0,
             response_body_bytes=600, expected_requests=3),
        dict(route_id="upload", method="POST", completed_requests=3, request_body_bytes=2100,
             response_body_bytes=50, expected_requests=1),
    ]
    expected = [420, 700, 130, 300, 4, 1, 2]
    assert operation_features(app) == expected
    payload = dict(contextual=True, separate_health=True, calibration_policy="v4", operation_context=True)
    assert context_width(payload) == 8
    assert feature_rows([row], FEATURES, True, True, False, True)[0][-8:].tolist() == [5, *expected]
    app["planned_activity"] = False
    app["operations"][0]["route_id"] = "changed"
    assert feature_rows([row], FEATURES, True, True, False, True)[0][-8:].tolist() == [5, *expected]
    app["completed_requests"] = 6
    assert operation_features(app) == [None] * 7


def operation(count, *, request=None, response=None, expected=None):
    return dict(route_id="upload", method="POST", completed_requests=count,
                request_body_bytes=request, response_body_bytes=response, expected_requests=expected)


@pytest.mark.parametrize("observations", [operation(4, request=0, expected=6), operation(5), None])
def test_operation_context_with_unobserved_features_keeps_only_http_decision(observations):
    row = window("test", 0, .5)
    if observations is not None:
        row["application"]["operations"] = [observations]
    row["application"]["server_errors"] = 5
    assert not operation_measurement_available(row["application"])
    assert not supported_inference_input(row, BINDING, "traffic-v1", True, True)
    bundle = dict(model_id="test", detector=dict(
        profile=PROFILE, feature_set_version="traffic-v1", binding=BINDING, contextual=True, operation_context=True,
        thresholds=dict(planned=None, unplanned=.5), application_limits=dict(min_requests=3, server_error_fraction=.2, latency_ms=100),
    ))
    result = validate("prediction", predict_window(row, bundle, independent_evidence=True))
    assert result["unavailable_reason"] == "application_context_missing"
    assert result["http"] == dict(status="anomalous", reason="error_fraction")
    row["application"]["operations"] = [operation(5, request=0)]
    assert operation_measurement_available(row["application"])
    assert supported_inference_input(row, BINDING, "traffic-v1", True, True)


def test_train_filters_network_only_and_requires_normal_operation_context(monkeypatch, tmp_path):
    train = [window("train", i, score) for i, score in enumerate((.3, .4, .8))]
    cal = [window("cal", i, score) for i, score in enumerate((.4, .9))]
    for w in (*train[:2], cal[0]):
        w["application"]["operations"] = [operation(5, request=0)]
    train[-1]["application"]["operations"] = [operation(5)]
    cal[-1]["application"].update(operations=[operation(4, request=0)], server_errors=2)
    parts = {"train": [(w, "normal") for w in train], "calibration": [(w, "normal") for w in cal]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(threshold_mode="incidents", feature_set_version="traffic-v1", binding=BINDING,
                                        separate_application_health=True, independent_application_evidence=True, operation_context=True)))
    monkeypatch.setattr(service_detector, "load_partitions", lambda *args: parts)
    from netwatch_ml.training import incident_calibration
    monkeypatch.setattr(incident_calibration, "calibrate_incidents", lambda *args, **kwargs: (
        dict(unplanned=.5, planned=None), dict(unsupported_regimes=["planned"]),
    ))
    report = service_detector.fit(manifest, tmp_path / "model", "isolation_forest", rules=True)
    assert report["network_training_windows"] == 2
    assert report["network_calibration_windows"] == 1
    assert report["http_normal_calibration_windows"] == 2
    assert report["application_limits"]["server_error_fraction"] == .5

    cal[0]["application"]["operations"] = [operation(5)]
    with pytest.raises(ValueError, match="normal training and calibration windows"):
        service_detector.fit(manifest, tmp_path / "unsupported", "isolation_forest", rules=True)
    assert not (tmp_path / "unsupported").exists()

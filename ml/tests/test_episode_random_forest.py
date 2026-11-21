import hashlib
import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from netwatch_ml.models.episode_classifier import (
    FEATURES, LABELS, REQUEST_RATIOS, forest_probabilities, load, probabilities,
)
from netwatch_ml.models.service_detector import APP_FEATURES
from netwatch_ml.training.random_forest_episode import class_thresholds, encode_forest, supported_thresholds


def test_portable_random_forest_matches_reference_probabilities(tmp_path):
    features = FEATURES + list(APP_FEATURES) + list(REQUEST_RATIOS)
    width = len(features) * 2
    rng = np.random.default_rng(12)
    raw = rng.uniform(0, 10, size=(120, width))
    values = np.column_stack((np.log1p(raw), np.zeros_like(raw)))
    labels = np.tile(np.arange(len(LABELS)), 20)
    classifier = RandomForestClassifier(n_estimators=8, max_depth=5, random_state=12).fit(values, labels)
    spec = encode_forest(classifier)
    np.testing.assert_allclose(forest_probabilities(values, spec), classifier.predict_proba(values), atol=1e-12)
    bundle = dict(schema_version=3, method="random_forest", labels=LABELS, features=features,
                  statistics=["median", "iqr"], binding={"sensor_id": "sensor", "entity_id": "TCP/127.0.0.1:8080"},
                  preprocessing={"median": [0] * width, "scale": [1] * width}, model=spec, temperature=1.,
                  application_limits=dict(min_requests=3, server_error_fraction=.2, latency_ms=100),
                  class_thresholds=dict.fromkeys(LABELS, .5))
    path = tmp_path / "model.json"
    path.write_text(json.dumps(bundle))
    saved = load(path, hashlib.sha256(path.read_bytes()).hexdigest())
    np.testing.assert_allclose(probabilities(raw, saved), classifier.predict_proba(values), atol=1e-12)


def test_thresholds_calibrate_each_forest_label_independently():
    scores = np.array([[.1, .8, .1, 0, 0, 0], [.1, .65, .25, 0, 0, 0],
                       [.1, .25, .55, .1, 0, 0], [.05, .85, .1, 0, 0, 0]])
    levels = class_thresholds(scores, np.array([1, 0, 2, 0]), np.array([False, False, False, True]))
    assert .65 < levels[LABELS[1]] < .8
    assert 0 < levels[LABELS[2]] < .55
    assert levels[LABELS[3]] > 1.


def test_version_five_requires_an_operating_point_across_distinct_captures():
    scores = np.array([[.1, .82, .08, 0, 0, 0], [.1, .81, .09, 0, 0, 0],
                       [.1, .20, .7, 0, 0, 0], [.1, .18, .72, 0, 0, 0],
                       [.1, .68, .22, 0, 0, 0], [.1, .08, .82, 0, 0, 0]])
    truth = np.array([1, 1, 2, 2, len(LABELS), len(LABELS)])
    eligible = np.ones(len(truth), dtype=bool)
    thresholds, support = supported_thresholds(scores, truth, np.array(list("AACDEF")), eligible)
    assert thresholds[LABELS[1]] > 1
    assert support[LABELS[1]]["status"] == "insufficient_capture_support"
    assert thresholds[LABELS[2]] > 1
    assert support[LABELS[2]]["status"] == "no_error_free_cutoff"
    assert support[LABELS[3]]["status"] == "missing_calibration_cases"
    thresholds, support = supported_thresholds(scores, truth, np.array(list("ABCDEF")), eligible)
    assert .68 < thresholds[LABELS[1]] < .81
    assert support[LABELS[1]]["accepted_captures"] == 2
    assert support[LABELS[1]]["precision"] == support[LABELS[1]]["coverage"] == 1.


def test_fits_v2_and_keeps_projection_labels_capture_local(tmp_path, monkeypatch):
    from netwatch_ml.models.detector import FEATURE_SETS, LIFECYCLE_VERSION
    from netwatch_ml.models.episode_classifier import UNKNOWN, WINDOW_NS, classify
    from netwatch_ml.training.random_forest_episode import fit

    version = LIFECYCLE_VERSION
    binding = {"sensor_id": "sensor", "entity_id": "TCP/127.0.0.1:8080"}

    def write(name, data):
        path = tmp_path / name
        path.write_text(data)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def captured(part, source):
        rows = []
        for phase, label in enumerate((*LABELS, UNKNOWN)):
            for j in range(3):
                i = 3 * phase + j
                window = dict(
                    **binding, schema_version=1, feature_set_version=version, event_id=f"{part}-{i}",
                    capture_epoch=part, entity_type="destination_service",
                    window_start_ns=str(i * WINDOW_NS), window_end_ns=str((i + 1) * WINDOW_NS),
                    features={**dict.fromkeys(FEATURE_SETS[version], None),
                              "new_tcp_attempts": phase + j + 1, "in_ip_bytes": (phase + 1) * 100,
                              "tcp_age_mean_seconds": float(phase + j)},
                    quality=dict(coverage="complete", capture_dropped=0, late_packets=0, state_evictions=0),
                    application=dict(completed_requests=3, server_errors=0, mean_latency_ms=2,
                                     planned_activity=False),
                )
                rows.append(dict(window=window, phase=phase, episode_label=label,
                                 label="normal" if phase == 0 else "anomaly"))
        name = f"{part}.jsonl"
        digest = write(name, "\n".join(json.dumps(row) for row in rows))
        capture = dict(source_sha256=source, windows=name, windows_sha256=digest)
        return rows, capture

    def projections(capture, name, *items):
        digest = write(name, json.dumps(dict(source_sha256=capture["source_sha256"], episodes=items)))
        capture["projected_episodes"] = {"path": name, "sha256": digest}

    train, train_meta = captured("train", "a" * 64)
    _, calibration_meta = captured("calibration", "b" * 64)
    normal = {"detector": "isolation_forest", "episode_label": "normal", "event_ids": ["train-0", "train-1"]}
    anomaly = {"detector": "mlp", "episode_label": LABELS[1], "event_ids": ["train-3", "train-4"]}
    projections(train_meta, "train-projected.json", normal, dict(normal, detector="mlp"), anomaly)
    projections(calibration_meta, "cal-projected.json", dict(normal, event_ids=["calibration-0", "calibration-1"]))
    definitions = dict(schema_version=1, feature_set_version=version, binding=binding,
                       captures={"train": [train_meta], "calibration": [calibration_meta]})
    location = tmp_path / "manifest.json"
    write(location.name, json.dumps(definitions))
    out = tmp_path / "out"
    report = fit(location, out)
    assert (report["projected_train_episodes"], report["projected_cal_episodes"]) == (2, 1)
    bundle = load(out / "random_forest.json", report["sha256"])
    assert bundle["schema_version"] == 4 and bundle["feature_set_version"] == version
    assert bundle["statistics"] == ["median", "iqr", "rise", "fall"]
    example = [row["window"] for row in train[:3]]
    assert classify(example, bundle)["label"] in (*LABELS, UNKNOWN)
    for window in example:
        window["feature_set_version"] = "traffic-v1"
    assert classify(example, bundle) == {"label": UNKNOWN, "reason": "insufficient_data"}

    foreign = dict(anomaly, event_ids=["calibration-3", "calibration-4"])
    projections(train_meta, "train-projected.json", foreign)
    write(location.name, json.dumps(definitions))
    with pytest.raises(ValueError, match="cross-capture window reference"):
        fit(location, tmp_path / "leaked")
    incorrect = dict(anomaly, episode_label=LABELS[2])
    projections(train_meta, "train-projected.json", incorrect)
    write(location.name, json.dumps(definitions))
    with pytest.raises(ValueError, match="conflicts with independent labels"):
        fit(location, tmp_path / "bad-labels")

    train, train_meta = captured("train", "a" * 64)
    extra_train, extra_train_meta = captured("train-extra", "c" * 64)
    calibration, calibration_meta = captured("calibration", "b" * 64)
    _, extra_cal_meta = captured("calibration-extra", "d" * 64)
    for row in calibration[-3:]:
        row["episode_evidence_channel"] = "network"
    calibration_meta["windows_sha256"] = write("calibration.jsonl", "\n".join(map(json.dumps, calibration)))
    operation = dict(route_id="status", method="GET", completed_requests=3,
                     request_body_bytes=0, response_body_bytes=300, expected_requests=3)
    for rows, capture in ((train, train_meta), (extra_train, extra_train_meta)):
        for row in rows:
            size = 900 if row["episode_label"] == "large_response" else 300
            row["window"]["application"]["operations"] = [dict(operation, response_body_bytes=size)]
        capture["windows_sha256"] = write(capture["windows"], "\n".join(map(json.dumps, rows)))
    definitions["captures"] = {"train": [train_meta, extra_train_meta], "calibration": [calibration_meta, extra_cal_meta]}
    definitions["episode_classifier_version"] = 5
    write(location.name, json.dumps(definitions))
    out = tmp_path / "v5"
    report = fit(location, out)
    bundle = load(out / "random_forest.json", report["sha256"])
    assert bundle["schema_version"] == 5 and bundle["operation_reference"] == {"GET status": [0., 100.]}
    assert bundle["operation_features"] == ["http_mean_response_body_bytes",
                                             "http_max_operation_mean_response_body_bytes",
                                             "response_body_excess_on_route"]
    assert report["network_open_set_calibration"]["episodes"] == 1
    assert report["unknown_without_network_evidence"] == 1
    assert report["http_unknown_calibration"]["episodes"] == 0
    missing_context = [dict(row["window"], application=None) for row in extra_train[:3]]
    assert classify(missing_context, bundle, evidence_channel="network").get("reason") != "insufficient_data"

    out = tmp_path / "omitted"
    report = fit(location, out, leave_out_type=LABELS[1])
    with pytest.raises(ValueError, match="experimental"):
        load(out / "random_forest.json", report["sha256"])
    bundle = load(out / "random_forest.json", report["sha256"], allow_experimental=True)
    assert bundle["class_thresholds"][LABELS[1]] > 1
    assert bundle["experimental_held_out_type"] == LABELS[1]
    assert report["class_support"][LABELS[1]]["positive"] == 0

    monkeypatch.setattr("netwatch_ml.models.episode_classifier.probabilities",
                        lambda *_: np.eye(len(LABELS))[:1])
    assert classify(missing_context, bundle, evidence_channel="network") == {
        "label": "normal", "reason": "normal_like_network_incident",
    }

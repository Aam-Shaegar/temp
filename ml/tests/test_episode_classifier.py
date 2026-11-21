import copy

import numpy as np
import pytest

from netwatch_ml.models.episode_classifier import (
    FEATURES, LABELS, classify, episode_features,
)
from netwatch_ml.training.episode_classification import abstention_threshold


def windows():
    return [
        {
            "sensor_id": "sensor",
            "entity_id": "TCP/127.0.0.1:8080",
            "capture_epoch": "run",
            "entity_type": "destination_service",
            "feature_set_version": "traffic-v1",
            "window_start_ns": i * 5_000_000_000,
            "window_end_ns": (i + 1) * 5_000_000_000,
            "quality": {
                "coverage": "complete",
                "capture_dropped": 0,
                "late_packets": 0,
                "state_evictions": 0,
            },
            "features": {name: value for name in FEATURES},
        }
        for i, value in enumerate((2, 6))
    ]


def test_features_preserve_missing_values_and_exclude_window_metadata():
    first, second = windows()
    first["features"][FEATURES[-1]] = second["features"][FEATURES[-1]] = None
    first["label"] = second["label"] = LABELS[1]
    result = episode_features([first, second])
    width = len(FEATURES)
    np.testing.assert_allclose(result[: width - 1], 4)
    np.testing.assert_allclose(result[width:-1], 2)
    assert np.isnan(result[width - 1]) and np.isnan(result[-1])


def test_unknown_for_loss_and_discontinuity():
    valid = windows()
    for key, value in (
        ("capture_epoch", "restarted"),
        ("window_start_ns", 10_000_000_000),
        ("quality", {**valid[1]["quality"], "capture_dropped": None}),
    ):
        sample = copy.deepcopy(valid)
        sample[1][key] = value
        assert classify(sample, {}) == {"label": "unknown", "reason": "insufficient_data"}


def test_abstention_preserves_class_recall_when_calibration_supports_it():
    truth = list(range(len(LABELS)))
    confidence = np.eye(len(LABELS)) * 0.6 + np.full((len(LABELS), len(LABELS)), 0.4 / len(LABELS))
    wrong = np.array([[0.20, 0.24, 0.22, 0.17, 0.13, 0.04]])
    probability = np.vstack([confidence, wrong])
    cutoff = abstention_threshold(probability, np.array([*truth, 0]))
    assert 0.24 < cutoff <= confidence.max()


def test_contextual_classifier_abstains_without_telemetry_and_on_application_failures():
    bundle = {
        "schema_version": 2,
        "binding": {"sensor_id": "sensor", "entity_id": "TCP/127.0.0.1:8080"},
        "application_limits": {"min_requests": 3, "server_error_fraction": 0.2, "latency_ms": 100},
    }
    rows = windows()
    assert classify(rows, bundle) == {"label": "unknown", "reason": "insufficient_data"}
    for row in rows:
        row["application"] = dict(completed_requests=10, server_errors=8, mean_latency_ms=2., planned_activity=True)
    assert classify(rows, bundle) == {"label": "unknown", "reason": "application_degradation"}
    rows[1]["application"]["planned_activity"] = None
    assert classify(rows, bundle) == {"label": "unknown", "reason": "insufficient_data"}


def test_journal_projection_excludes_recovery_and_abstains_on_partial_evidence():
    from netwatch_ml.models.episode_classifier import journal_windows, merge_journal_pages

    sample = windows()
    events = [
        dict(event=f"window-{i}", start_ns=str(i * 5_000_000_000), end_ns=str((i+1) * 5_000_000_000),
             features=sample[i % 2]["features"], quality=sample[i % 2]["quality"], application=None)
        for i in range(3)
    ]
    incident = dict(id="173", first_ns="0", estimated_end_ns="10000000000", sensor="sensor", epoch="run",
                    entity="TCP/127.0.0.1:8080", entity_type="destination_service")
    journal = dict(incident=incident, feature_set_version="traffic-v1", events=events)
    chosen = journal_windows(journal)
    assert [w["event_id"] for w in chosen] == ["window-0", "window-1"]
    assert all("application" not in w for w in chosen)
    assert journal_windows(dict(journal, events=events[:1])) == []
    first = dict(journal, events=events[:1], more=True)
    second = dict(journal, events=events[1:], more=False)
    assert journal_windows(merge_journal_pages([first, second])) == chosen
    with pytest.raises(ValueError, match="different incident"):
        merge_journal_pages([first, dict(second, incident=dict(incident, id="37"))])
    with pytest.raises(ValueError, match="repeat a window"):
        merge_journal_pages([first, dict(second, events=events)])
    incident["estimated_end_ns"] = None
    assert journal_windows(journal) == []


def test_v2_preserves_missing_lifecycle_data_and_differs_by_temporal_direction():
    from netwatch_ml.models.detector import FEATURE_SETS, LIFECYCLE_VERSION

    rows = windows()
    names = FEATURE_SETS[LIFECYCLE_VERSION]
    for index, row in enumerate(rows):
        row["feature_set_version"] = LIFECYCLE_VERSION
        age = {"tcp_age_mean_seconds": (2., 9.)[index],
               "tcp_age_max_seconds": (3., 11.)[index],
               "tcp_age_known_start_fraction": (0.5, 1.)[index]}
        row["features"].update({key: age.get(key) for key in names if key not in row["features"]})
    result = episode_features(rows, feature_set_version=LIFECYCLE_VERSION).reshape(4, -1)
    age = names.index("tcp_age_mean_seconds")
    censored = names.index("tcp_closed_duration_mean_seconds")
    assert result[2, age] == 7. and result[3, age] == 0.
    assert np.isnan(result[:, censored]).all()

    rows[0]["features"]["tcp_age_mean_seconds"] = 9.
    rows[1]["features"]["tcp_age_mean_seconds"] = 2.
    reversed_trend = episode_features(rows, feature_set_version=LIFECYCLE_VERSION).reshape(4, -1)
    assert reversed_trend[0, age] == result[0, age]
    assert reversed_trend[2, age] == 0. and reversed_trend[3, age] == 7.
    assert episode_features(rows) is None

    rows[1]["quality"]["coverage"] = "partial"
    assert episode_features(rows, feature_set_version=LIFECYCLE_VERSION) is None


def test_http_channel_classifies_closed_degradation_without_network_coverage():
    bundle = {
        "schema_version": 2,
        "binding": {"sensor_id": "sensor", "entity_id": "TCP/127.0.0.1:8080"},
        "application_limits": {"min_requests": 3, "server_error_fraction": 0.2, "latency_ms": 100},
    }
    rows = windows()
    for row in rows:
        row["quality"]["coverage"] = "gap"
        row["application"] = dict(completed_requests=10, server_errors=8, mean_latency_ms=2., planned_activity=None)
    assert classify(rows, bundle, evidence_channel="http") == {"label": "unknown", "reason": "application_degradation"}
    assert classify(rows, bundle, evidence_channel="network") == {"label": "unknown", "reason": "insufficient_data"}
    rows[1]["capture_epoch"] = "different-epoch"
    assert classify(rows, bundle, evidence_channel="http") == {"label": "unknown", "reason": "insufficient_data"}

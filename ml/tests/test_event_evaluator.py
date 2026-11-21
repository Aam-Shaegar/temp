"""Regression checks for the shared event reducer and the acceptance bridge."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks import detection_acceptance, event_evaluator  # noqa: E402

NS = event_evaluator.NS


def truth(start, end, label, **attributes):
    return dict(start=start * NS, end=end * NS, label=label, **attributes)


def incident(id, first, end, detected=None, **attributes):
    return dict(id=id, first=first * NS, end=end * NS, last=end * NS,
                detected=(first if detected is None else detected) * NS, **attributes)


def observation(normal_seconds, *, windows=10, evaluated=10, **attributes):
    return dict(windows=windows, evaluated=evaluated, normal_evaluated_ns=normal_seconds * NS, **attributes)


def group(id, truth_rows, incidents, observed=None, *, service="service-a"):
    return dict(id=id, service=service, truth=truth_rows, incidents=incidents, observation=observed)


def test_maximum_pairs_reassign_earlier_largest_overlap():
    source = group("capture", [truth(0, 10, "surge"), truth(10, 20, "surge")],
                   [incident("small", 0, 4), incident("largest", 5, 13)])
    result = event_evaluator.evaluate_groups([source])
    actual = result["cases"][0]
    assert [actual["incidents"][p["match"]]["id"] for p in actual["positive"]] == ["small", "largest"]
    assert result["overall"]["known_found"] == 2
    assert result["overall"]["split"] == result["overall"]["merged"] == 1
    assert result["overall"]["checks"]["episode_integrity"] is False


def test_maximum_cardinality_prefers_larger_total_overlap():
    edges = [(0, 0, 5), (0, 1, 7), (1, 0, 8), (1, 1, 2)]
    assert event_evaluator.maximum_matches(2, 2, edges) == {0: 1, 1: 0}


def test_legacy_matching_uses_confirmation_and_scoped_matching_uses_first_evidence():
    source = group("case", [truth(10, 20, "surge")], [incident("late", 8, 22, 20)])
    legacy = event_evaluator.evaluate_groups([source], start_tolerance_s=None)
    timed = event_evaluator.evaluate_groups([source], start_tolerance_s=5)
    assert legacy["overall"]["known_found"] == 0
    assert timed["overall"]["known_found"] == 1
    assert timed["overall"]["start_error_s"]["max"] == 2
    assert timed["overall"]["opening_delay_s"]["max"] == 10


def test_separate_channels_and_unknown_outcomes():
    reference = [
        truth(0, 10, "surge", evidence_channel="network"),
        truth(0, 10, "unknown", evidence_channel="http"),
        truth(20, 30, "unknown", evidence_channel="network"),
        truth(40, 50, "unknown", evidence_channel="network"),
        truth(60, 70, "unknown", evidence_channel="network"),
        truth(80, 90, "mixed", evidence_channel="network"),
    ]
    incidents = [
        incident("http", 0, 10, evidence_channel="http",
                 outputs={"rf": {"label": "unknown", "reason": "application_degradation"}}),
        incident("network", 0, 10, evidence_channel="network", outputs={"rf": {"label": "surge"}}),
        incident("unsupported", 20, 30, evidence_channel="network", outputs={"rf": {"label": "surge"}}),
        incident("bad-reason", 40, 50, evidence_channel="network",
                 outputs={"rf": {"label": "unknown", "reason": "application_degradation"}}),
        incident("mixed", 80, 90, evidence_channel="network",
                 outputs={"rf": {"label": "unknown", "reason": "low_confidence"}}),
    ]
    result = event_evaluator.evaluate_groups([group("case", reference, incidents)], classifier="rf")
    score = result["overall"]["classifier"]
    assert result["known_labels"] == ("surge",)
    assert result["overall"]["known_found"] == 1
    assert result["overall"]["unknown_found"] == 3
    assert result["overall"]["split"] == result["overall"]["merged"] == 0
    assert score["unknown_total"] == 4 and score["explicit_unknown"] == 1
    assert score["unknown_overclaimed"] == 1
    assert score["unknown_outcomes"]["not_detected"] == 1
    assert score["mixed_truth_outcomes"] == {"unknown:low_confidence": 1}
    assert score["unknown_by_channel"]["http"] == {"total": 1, "rejected": 1, "recall": 1}
    assert score["unknown_by_channel"]["network"] == {"total": 3, "rejected": 0, "recall": 0}


def test_http_only_refusals_and_unplanned_only_network_cannot_meet_explicit_support_gates():
    reference = [truth(0, 10, "surge", planned=False, evidence_channel="network"),
                 truth(20, 30, "unknown", planned=True, evidence_channel="http")]
    incidents = [incident("http", 20, 30, evidence_channel="http",
                          outputs={"rf": {"label": "unknown", "reason": "application_degradation"}})]
    result = event_evaluator.evaluate_groups(
        [group("case", reference, incidents)],
        {"require_network_unknown": True, "min_known_by_regime": {"planned": 1}}, classifier="rf",
    )["overall"]
    assert result["checks"]["planned_network_support"] is False
    assert result["checks"]["planned_network_recall"] is False
    assert result["known_by_regime"]["planned"]["recall"] is None
    assert result["classifier"]["gate"]["unknown"] is True
    assert result["classifier"]["gate"]["network_unknown"] is False
    assert result["classifier"]["unknown_by_channel"]["network"]["recall"] is None


def test_regime_with_enough_examples_still_fails_if_all_its_episodes_are_missed():
    reference = [truth(start, start + 10, "surge", planned=(start == 0), evidence_channel="network")
                 for start in (0, 20, 40, 60, 80)]
    incidents = [incident(str(start), start, start + 10, evidence_channel="network") for start in (20, 40, 60, 80)]
    result = event_evaluator.evaluate_groups([group("case", reference, incidents)],
        {"min_known_by_regime": {"planned": 1, "unplanned": 1}}, known_labels=("surge",))["overall"]
    assert result["checks"]["recall"] is True
    assert result["checks"]["planned_network_support"] is True
    assert result["checks"]["planned_network_recall"] is False
    assert result["checks"]["unplanned_network_recall"] is True


def test_absolute_timing_and_unresolved_close_gate_are_separate():
    g = group("case", [truth(10, 20, "surge")],
              [dict(incident("partial", 10, 19, 19), end=None, interrupted=19 * NS)])
    result = event_evaluator.evaluate_groups([g])["overall"]
    assert result["checks"]["start_error"] is True
    assert result["opening_delay_s"]["mean"] == 9
    assert result["checks"]["end_error"] is False and result["unresolved_ends"] == 1


def test_false_alarm_confirmation_planned_strata_and_service_counts_use_one_reducer():
    reference = [truth(0, 10, "normal", planned=True), truth(10, 20, "surge", evidence_channel="network"),
                 truth(20, 40, "normal", planned=False)]
    g = group("a", reference, [
        incident("plain", 0, 2, 1, evidence_channel="network"),
        incident("starts-before", 3, 15, 5, evidence_channel="network"),
        incident("valid", 10, 20, 15, evidence_channel="network"),
    ], observation(24, windows=5, evaluated=4,
                   normal_evaluated_ns_by_regime={"planned": 8 * NS, "unplanned": 16 * NS}))
    g2 = group("b", [truth(0, 10, "normal")], [incident("unplanned", 0, 5, 2, evidence_channel="http")],
               observation(10), service="service-b")
    result = event_evaluator.evaluate_groups([g, g2])
    whole = result["overall"]
    assert whole["false_alarms"] == sum(service["false_alarms"] for service in result["by_service"].values()) == 3
    assert whole["pure_false_alarms"] == 2
    assert whole["false_alarms_by_evidence_channel"] == {"network": 2, "http": 1}
    assert whole["by_normal_regime"]["planned"]["false_alarms"] == 2
    assert whole["by_normal_regime"]["planned"]["normal_evaluated_hours"] == 8 / 3600
    assert whole["by_normal_regime"]["unspecified"]["normal_evaluated_hours"] == 10 / 3600
    assert whole["normal_evaluated_hours"] == 34 / 3600
    assert whole["evaluated_fraction"] == 14 / 15


def test_labelled_hours_only_cannot_pass_normal_exposure_and_missing_classes_are_visible():
    g = group("archived", [truth(0, 36000, "normal", planned=True), truth(36000, 36010, "surge")], [
        incident("valid", 36000, 36010, outputs={"rf": {"label": "surge"}}),
    ])
    result = event_evaluator.evaluate_groups(
        [g], {"min_episodes_per_known_class": 1}, known_labels=("surge", "pause"), classifier="rf",
    )["overall"]
    assert result["normal_labelled_hours"] == 10
    assert result["normal_evaluated_hours"] is None and result["normal_hours_kind"] == "labelled_only"
    assert result["by_normal_regime"]["planned"]["false_per_hour"] is None
    assert result["checks"]["normal_exposure"] is False and result["checks"]["known_support"] is False
    assert result["classifier"]["macro_recall"] is None


def test_manifest_annotations_legacy_and_channel_aware_formats():
    assert detection_acceptance.read_annotations({"anomalous": [["10", "20"]], "normal": [[0, 10]]}, 0, 20) == [
        dict(start=0, end=10, label="normal"), dict(start=10, end=20, label="anomalous"),
    ]
    rows = [truth(0, 10, "normal", planned=True), truth(10, 20, "surge", evidence_channel="network")]
    assert detection_acceptance.read_annotations({"truth": rows}, 0, 20 * NS) == rows
    with pytest.raises(ValueError, match="Choose one"):
        detection_acceptance.read_annotations({"truth": rows, "normal": []}, 0, 20 * NS)
    with pytest.raises(ValueError, match="outside"):
        detection_acceptance.read_annotations({"truth": rows}, 0, 15 * NS)


def test_live_acceptance_retains_http_channel_and_legacy_fields():
    def timestamp(seconds):
        return str(seconds * NS)

    windows = [
        dict(event_id=str(i), window_start_ns=timestamp(i), window_end_ns=timestamp(i + 5),
             entity_id="service", capture_epoch="capture", feature_set_version="traffic-v2")
        for i in range(0, 25, 5)
    ]
    events = [
        dict(id="h", epoch="capture", evidence_channel="http", first_ns=timestamp(10), detected_ns=timestamp(12),
             last_ns=timestamp(16), estimated_end_ns=timestamp(16), closed_ns=timestamp(17)),
        dict(id="n", epoch="capture", evidence_channel="network", first_ns=timestamp(10), detected_ns=timestamp(13),
             last_ns=timestamp(17), estimated_end_ns=timestamp(17), closed_ns=timestamp(18)),
    ]

    def call(path, *_):
        if path == "/v1/windows":
            body = {}
        elif path.startswith("/v1/incidents?"):
            body = dict(pending=False, incidents=events, next="")
        elif path.startswith("/v1/incidents/"):
            body = dict(feature_set_version="traffic-v2")
        elif path.startswith("/v1/predictions/"):
            body = dict(predictions=[dict(prediction=dict(status="normal"))])
        else:
            pytest.fail(f"Unexpected route {path}")
        return 200, {}, json.dumps(body)

    annotations = [truth(0, 10, "normal", planned=True), truth(10, 20, "surge", evidence_channel="network")]
    report = detection_acceptance.evaluate(call, "sensor", "operator", {"start_tolerance_s": 5},
                                           [("capture", windows, annotations)])
    assert report["totals"]["positive"] == report["totals"]["found"] == 1
    assert report["cases"][0]["episodes"][0]["opening_delay_s"] == 3
    assert report["by_service"]["service"]["by_normal_regime"]["planned"]["normal_evaluated_hours"] == 10 / 3600
    assert report["overall"]["split"] == 0

"""Calibrate service scores by replaying continuous, separately labelled captures."""

from collections import Counter
from itertools import islice, product
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.event_evaluator import NON_KNOWN, evaluate_groups, overlap, reference_intervals
from netwatch_ml.contracts import validate
from netwatch_ml.models.application_operations import operation_measurement_available
from netwatch_ml.models.detector import supported_window
from netwatch_ml.models.service_detector import INCIDENT_CALIBRATION_POLICY, application_available, planned_state
from netwatch_ml.training.baselines import BUDGET, WINDOW_NS, sha

MIN_REGIME_CAPTURES = 2
START_TOLERANCE_S = 5
CALIBRATION_LIMITS = dict(max_false_per_hour=BUDGET)


def load_calibration_timeline(capture, root, binding, version, rows):
    source_path = Path(capture["timeline"])
    path = (root / source_path).resolve()
    if (source_path.is_absolute() or not path.is_relative_to(root.resolve())
            or sha(path) != capture["timeline_sha256"]):
        raise ValueError("invalid incident calibration timeline provenance")
    stored = json.loads(path.read_text())
    if stored["source_sha256"] != capture["source_sha256"] or not stored["windows"]:
        raise ValueError("calibration timeline differs from its capture")
    if not rows:
        raise ValueError("empty incident calibration capture")
    identity = rows[0][0]["capture_epoch"]
    labelled = {w["event_id"]: w for w, _ in rows}
    seen = set()
    previous = None
    for window in stored["windows"]:
        validate("window", window)
        start, end = int(window["window_start_ns"]), int(window["window_end_ns"])
        event = window["event_id"]
        if (window["capture_epoch"] != identity or window["entity_type"] != "destination_service"
                or window["feature_set_version"] != version or any(window[k] != v for k, v in binding.items())
                or event in seen or start % WINDOW_NS or end - start != WINDOW_NS
                or (previous is not None and start != previous)
                or (event in labelled and labelled.pop(event) != window)):
            raise ValueError("expected continuous ordered windows from one calibration capture")
        seen.add(event)
        previous = end
    if labelled:
        raise ValueError("calibration timeline omits labelled windows")
    begin, finish = int(stored["windows"][0]["window_start_ns"]), previous
    truth = stored["truth"]
    reference_intervals(dict(truth=truth))
    if not truth or any(row["start"] < begin or row["end"] > finish for row in truth):
        raise ValueError("missing calibration truth or truth outside the capture")
    # Health is a separate v4 input. Preserve its windows, but reserve their
    # annotations for HTTP evaluation rather than treating them as network truth.
    relevant = [dict(row, label="unobservable") if row.get("evidence_channel") == "http" else row for row in truth]
    if any(type(row.get("planned")) is not bool for row in relevant if row["label"] != "unobservable"):
        raise ValueError("incident calibration truth requires an explicit schedule state")
    if any(row.get("evidence_channel") != "network" for row in relevant if row["label"] not in NON_KNOWN):
        raise ValueError("incident calibration requires independently labelled network intervals")
    for window, label in rows:
        start, end = int(window["window_start_ns"]), int(window["window_end_ns"])
        if not any(row["start"] <= start and end <= row["end"]
                   and (row["label"] == "normal") == (label == "normal") for row in truth):
            raise ValueError("calibration window labels differ from continuous truth")
    return dict(id=identity, service=binding["entity_id"], windows=stored["windows"], truth=relevant)


def supported_inference_input(window, binding, version, contextual, operation_context=False):
    if not supported_window(window, binding, version):
        return False
    if contextual and not application_available(window):
        return False
    if operation_context and not operation_measurement_available(window["application"]):
        return False
    return True


def calibration_inputs(timelines, score, binding, version, contextual, operation_context=False):
    groups = []
    for timeline in timelines:
        rows = [(window, planned_state(window, contextual)) for window in timeline["windows"]
                if supported_inference_input(window, binding, version, contextual, operation_context)]
        values = score([w for w, _ in rows]) if rows else ()
        scores = {w["event_id"]: (state, float(value)) for (w, state), value in zip(rows, values, strict=True)}
        points = [(dict(ID=w["event_id"], Start=int(w["window_start_ns"]), End=int(w["window_end_ns"])),
                   scores.get(w["event_id"])) for w in timeline["windows"]]
        groups.append(dict(id=timeline["id"], service=timeline["service"], truth=timeline["truth"], points=points))
    return groups


def regime_support(groups, regime, contextual):
    reports = Counter()
    for group in groups:
        complete = [(p["Start"], p["End"]) for p, score in group["points"] if score is not None and score[0] == regime]
        normal = episodes = 0
        for row in group["truth"]:
            if row["label"] == "unobservable" or (contextual and row["planned"] != (regime == "planned")):
                continue
            full = [start for start, end in complete if row["start"] <= start and end <= row["end"]]
            if row["label"] == "normal":
                normal += len(full)
            elif row["label"] not in NON_KNOWN:
                episodes += any(b - a == WINDOW_NS for a, b in zip(full, full[1:]))
        reports["normal_windows"] += normal
        reports["normal_captures"] += bool(normal)
        reports["positive_episodes"] += episodes
        reports["positive_captures"] += bool(episodes)
    return dict(reports)


def replay_input(group, thresholds, contextual):
    points = []
    normal = Counter()
    evaluated = 0
    for bounds, score in group["points"]:
        level = thresholds[score[0]] if score is not None else None
        if level is None:
            points.append(dict(bounds, Status="insufficient_data"))
            continue
        value = score[1]
        evaluated += 1
        points.append(dict(bounds, Score=value, Threshold=level, Status="anomalous" if value > level else "normal"))
        for row in group["truth"]:
            if row["label"] == "normal":
                regime = ("planned" if row["planned"] else "unplanned") if contextual else "unplanned"
                if regime == score[0]:
                    normal[regime] += overlap(bounds["Start"], bounds["End"], row["start"], row["end"])
    truth = group["truth"] if contextual else [dict(row, planned=False) for row in group["truth"]]
    report = dict(id=group["id"], service=group["service"], truth=truth,
                  observation=dict(windows=len(points), evaluated=evaluated, normal_evaluated_ns=sum(normal.values()),
                                   normal_evaluated_ns_by_regime=dict(normal)))
    return dict(Policy=INCIDENT_CALIBRATION_POLICY, Points=points), report


def candidate_thresholds(groups, regime):
    scores = [score[1] for group in groups for _, score in group["points"] if score is not None and score[0] == regime]
    levels = set(float(v) for v in np.quantile(scores, np.linspace(0, 1, 17)) if v < 1.)
    levels.update(float(np.nextafter(v, 0.)) for v in tuple(levels) + (max(scores),) if v > 0.)
    return sorted({0., *levels})


def replay_candidates(groups, candidates, projector, contextual):
    remaining = iter(candidates)
    while batch := list(islice(remaining, 8)):
        prepared = [[replay_input(group, candidate, contextual) for group in groups] for candidate in batch]
        payload = [replay for entries in prepared for replay, _ in entries]
        try:
            run = subprocess.run([str(projector)], input=json.dumps(payload, allow_nan=False), text=True,
                                 stdout=subprocess.PIPE, check=True, timeout=120)
        except FileNotFoundError as exc:
            raise ValueError("incident calibration requires the netwatch-projection-driver executable") from exc
        results = iter(json.loads(run.stdout))
        for candidate, entries in zip(batch, prepared, strict=True):
            scored = [dict(entry, incidents=next(results)) for _, entry in entries]
            report = evaluate_groups(scored, CALIBRATION_LIMITS, start_tolerance_s=START_TOLERANCE_S)
            yield candidate, report


def selection_key(candidate, metrics):
    overall = metrics["overall"]
    gate = overall["checks"]
    boundary_penalty = sum(not gate[key] for key in ("start_error", "end_error", "episode_integrity"))
    false_rate = overall["false_per_hour"]
    meets_targets = gate["recall"] and boundary_penalty == 0
    errors = (overall["start_error_s"]["max"] or 0.) + (overall["end_error_s"]["max"] or 0.)
    return (false_rate is None or false_rate > BUDGET, not meets_targets, -overall["known_found"],
            boundary_penalty, overall["false_alarms"], errors, -sum(v for v in candidate.values() if v is not None))


def calibrate_incidents(timelines, score, *, binding, version, contextual, operation_context=False,
                        projector="netwatch-projection-driver"):
    """Select epoch thresholds with event-time matches and explicit support requirements."""
    groups = calibration_inputs(timelines, score, binding, version, contextual, operation_context)
    regimes = ("unplanned", "planned") if contextual else ("unplanned",)
    support = {regime: regime_support(groups, regime, contextual) for regime in regimes}
    supported = {regime for regime, counts in support.items()
                 if min(counts["positive_captures"], counts["normal_captures"]) >= MIN_REGIME_CAPTURES}
    if not supported:
        raise ValueError(f"incident calibration has no supported network regime: {support}")
    levels = [candidate_thresholds(groups, regime) if regime in supported else [None] for regime in regimes]
    candidates = (dict(zip(regimes, values, strict=True)) for values in product(*levels))
    best, report = min(replay_candidates(groups, candidates, projector, contextual), key=lambda item: selection_key(*item))
    return best, dict(policy=INCIDENT_CALIBRATION_POLICY, candidate_selection="continuous_journal", support=support,
                      min_regime_calibration_captures=MIN_REGIME_CAPTURES,
                      unsupported_regimes=sorted(set(regimes) - supported), limits=report["limits"],
                      metrics=report["overall"])

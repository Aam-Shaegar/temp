"""Train and calibrate portable detectors using disjoint labeled service captures."""

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from netwatch_ml.contracts import validate
from netwatch_ml.models.application_operations import operation_measurement_available
from netwatch_ml.models.detector import FEATURE_SETS, POLICY, supported_window, transform, validate_binding
from netwatch_ml.models.service_detector import (
    INCIDENT_CALIBRATION_POLICY, PROFILE, application_available, degradation, feature_rows, model_scores, planned_state, train_scaler,
    validate_application_limits,
)
from netwatch_ml.training.baselines import BUDGET, WINDOW_NS, calibrate, fit_forest, publish_bundle, sha

METHODS = ("isolation_forest", "mlp")
MLP_MAX_TRAIN_DEVIATION = 32.
LEGACY_MLP_MAX_ITERATIONS = 1000
INCIDENT_MLP_MAX_ITERATIONS = 2000


def load_partitions(manifest, root, contextual, timelines=None):
    binding = manifest["binding"]
    validate_binding(binding)
    version = manifest["feature_set_version"]
    if manifest["schema_version"] != 1 or version not in FEATURE_SETS or set(manifest["captures"]) != {"train", "calibration"}:
        raise ValueError("invalid service calibration manifest")
    parts = {}
    hashes, epochs, ids, starts = set(), set(), set(), set()
    for part, captures in manifest["captures"].items():
        rows = []
        for capture in captures:
            source = capture["source_sha256"]
            path = (root / capture["windows"]).resolve()
            if (not re.fullmatch("[a-f0-9]{64}", source) or source in hashes
                    or Path(capture["windows"]).is_absolute() or not path.is_relative_to(root.resolve())
                    or sha(path) != capture["windows_sha256"]):
                raise ValueError("invalid or repeated capture")
            hashes.add(source)
            files = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            capture_epochs = {row["window"]["capture_epoch"] for row in files}
            if len(capture_epochs) != 1 or epochs & capture_epochs:
                raise ValueError("capture must have a unique observation epoch")
            epochs.update(capture_epochs)
            capture_rows = []
            for row in files:
                window = validate("window", row["window"])
                start = int(window["window_start_ns"])
                position = (window["capture_epoch"], start)
                if (window["event_id"] in ids or position in starts or start % 5_000_000_000
                        or row["label"] not in ("normal", "anomaly") or not supported_window(window, binding, version)
                        or contextual and not application_available(window)):
                    raise ValueError("invalid labeled service window")
                ids.add(window["event_id"])
                starts.add(position)
                capture_rows.append((window, row["label"]))
            if timelines is not None and part == "calibration":
                from netwatch_ml.training.incident_calibration import load_calibration_timeline
                timelines.append(load_calibration_timeline(capture, root, binding, version, capture_rows))
            rows.extend(capture_rows)
        if not rows or not any(label == "normal" for _, label in rows):
            raise ValueError(f"{part} requires labeled service data containing normal windows")
        parts[part] = rows
    return parts


def application_limits(normal, configured=None):
    """Conservative initial limits for services without a supplied latency SLO."""
    if configured is not None:
        validate_application_limits(configured)
        return configured
    samples = [w["application"] for w in normal if w["application"]["completed_requests"] >= 3]
    if not samples:
        raise ValueError("application rule requires normal HTTP observations")
    latency = 3 * float(np.quantile([a["mean_latency_ms"] for a in samples], .99))
    errors = max(a["server_errors"] / a["completed_requests"] for a in samples)
    if errors >= .9:
        raise ValueError("normal observations have too many server errors to calibrate the rule")
    return dict(min_requests=3, latency_ms=max(40., latency),
                server_error_fraction=max(.2, errors + .1))


def confirmed_pair_thresholds(rows, scores, contextual, limits):
    """Calibrate to contiguous pairs, the evidence required to open a v3 alert.

    Normal pairs are a conservative proxy for false openings. Full incident
    acceptance on locked continuous captures is a separate measurement.
    """
    data = list(zip(rows, scores, strict=True))
    data.sort(key=lambda row: (row[0][0]["capture_epoch"], int(row[0][0]["window_start_ns"])))
    states = ("unplanned", "planned") if contextual else ("unplanned",)
    by_state = {state: [] for state in states}
    inputs = []
    last_window, last_label, phase = None, None, 0
    normal_windows = 0
    for (window, label), score in data:
        state = planned_state(window, contextual)
        by_state[state].append(float(score))
        consecutive = (last_window is not None and window["capture_epoch"] == last_window["capture_epoch"]
                       and window["window_start_ns"] == last_window["window_end_ns"] and label == last_label)
        if not consecutive:
            phase += 1
        inputs.append((phase, state, label, float(score), degradation(window, limits)))
        normal_windows += label == "normal"
        last_label, last_window = label, window

    candidates = [[float(x) for x in np.unique(np.r_[0., np.quantile(by_state[s], np.linspace(0, 1, 17)), 1.])]
                  for s in states]
    budget = int(BUDGET * normal_windows * WINDOW_NS / 3_600_000_000_000)
    best = None
    for levels in product(*candidates):
        thresholds = dict(zip(states, levels, strict=True))
        found = set()
        false_pairs = 0
        previous_phase, previous_signal = None, False
        for phase, state, label, score, unhealthy in inputs:
            signal = unhealthy or score > thresholds[state]
            if signal and previous_signal and phase == previous_phase:
                if label == "normal":
                    false_pairs += 1
                else:
                    found.add(phase)
            previous_phase, previous_signal = phase, signal
        key = (max(false_pairs - budget, 0), -len(found), false_pairs, -sum(levels), tuple(-x for x in levels))
        if best is None or key < best[0]:
            best = (key, thresholds, dict(false_normal_pairs=false_pairs, positive_runs_with_pair=len(found)))
    return best[1:]


def bound_mlp_train_scale(raw, params):
    """Retain the healthy median and widen tiny IQRs using only supervised train rows."""
    center = np.asarray(params["median"])
    delta = np.abs(np.log1p(raw) - center)
    radius = np.max(np.where(np.isfinite(delta), delta, 0.), axis=0)
    scale = np.maximum(params["scale"], radius / MLP_MAX_TRAIN_DEVIATION)
    return dict(params, scale=scale.tolist())


def fit(manifest_path, destination, method, *, contextual=True, rules=False, seed=42,
        projector="netwatch-projection-driver"):
    if rules and not contextual:
        raise ValueError("application rules require context")
    if method not in METHODS:
        raise ValueError("invalid service detector")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    version = manifest["feature_set_version"]
    features = FEATURE_SETS[version]
    independent_application = manifest.get("independent_application_evidence", False)
    if type(independent_application) is not bool or (independent_application and not rules):
        raise ValueError("independent application evidence requires an application rule")
    mode = manifest.get("threshold_mode", "windows")
    if mode not in ("windows", "confirmed_pairs", "incidents"):
        raise ValueError("unsupported threshold calibration mode")
    separate_health = manifest.get("separate_application_health", False)
    if type(separate_health) is not bool or (separate_health and (not contextual or not rules)):
        raise ValueError("separate application health requires the independent application rule")
    incident_calibration = mode == "incidents"
    operation_context = manifest.get("operation_context", False)
    if type(operation_context) is not bool or (operation_context and not (contextual and incident_calibration)):
        raise ValueError("operation context requires contextual incident calibration")
    if incident_calibration and ((contextual and not separate_health) or (rules and not independent_application)):
        raise ValueError("incident calibration requires independent HTTP rules for contextual models")
    timelines = [] if incident_calibration else None
    parts = load_partitions(manifest, manifest_path.parent, contextual, timelines)
    http_normal_cal = [w for w, label in parts["calibration"] if label == "normal"]
    if operation_context:
        parts = {part: [(w, label) for w, label in rows if operation_measurement_available(w["application"])]
                 for part, rows in parts.items()}
    normal_train = [w for w, label in parts["train"] if label == "normal"]
    normal_cal = [w for w, label in parts["calibration"] if label == "normal"]
    if operation_context and (not normal_train or not normal_cal):
        raise ValueError("operation context must be measured in normal training and calibration windows")
    feature_arguments = (features, contextual, separate_health, not incident_calibration, operation_context)
    params = train_scaler(normal_train, *feature_arguments)

    def design(rows):
        return transform(feature_rows(rows, *feature_arguments), params)

    fit_details = {}
    if method == "isolation_forest":
        _, forest = fit_forest(design(normal_train), seed)
        learned = {"forest": forest}
    else:
        raw_train = feature_rows([w for w, _ in parts["train"]], *feature_arguments)
        params = bound_mlp_train_scale(raw_train, params)
        x = transform(raw_train, params)
        y = np.asarray([label == "anomaly" for _, label in parts["train"]], dtype=int)
        if len(set(y)) != 2:
            raise ValueError("MLP requires labeled anomalies in training")
        # Removing the schedule shortcut makes the supervised objective harder.
        # The bounded event-mode optimizer must still converge before export.
        max_iter = INCIDENT_MLP_MAX_ITERATIONS if incident_calibration else LEGACY_MLP_MAX_ITERATIONS
        model = MLPClassifier(hidden_layer_sizes=(16,), solver="lbfgs", alpha=.1, max_iter=max_iter, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(x, y)
            except ConvergenceWarning as exc:
                raise ValueError("MLP did not converge; no detector artifact published") from exc
        fit_details = dict(mlp_iterations=int(model.n_iter_),
                           mlp_max_train_deviation_limit=MLP_MAX_TRAIN_DEVIATION,
                           mlp_max_train_deviation_observed=float(np.max(np.abs(x))))
        if incident_calibration:
            fit_details["mlp_max_iterations"] = max_iter
        learned = {"layers": [[w.tolist(), b.tolist()] for w, b in zip(model.coefs_, model.intercepts_, strict=True)]}
        np.testing.assert_allclose(model_scores(x, dict(method=method, **learned)), model.predict_proba(x)[:, 1], atol=1e-12)
    scores = model_scores(design(normal_cal), dict(method=method, **learned))
    states = np.asarray([planned_state(w, contextual) for w in normal_cal])
    required_states = ("unplanned", "planned") if contextual else ("unplanned",)
    limits = application_limits(http_normal_cal, manifest.get("application_limits")) if rules else None
    calibration = {}
    if incident_calibration:
        from netwatch_ml.training.incident_calibration import calibrate_incidents

        def recorded_scores(windows):
            return model_scores(design(windows), dict(method=method, **learned))

        thresholds, calibration = calibrate_incidents(timelines, recorded_scores, binding=manifest["binding"],
                                                      version=version, contextual=contextual,
                                                      operation_context=operation_context, projector=projector)
    else:
        if set(states) != set(required_states):
            raise ValueError("calibration needs normal examples for every schedule state")
        thresholds = {state: calibrate(scores[states == state]) for state in required_states}
    if mode == "confirmed_pairs":
        cal_rows = parts["calibration"]
        thresholds, calibration = confirmed_pair_thresholds(
            cal_rows, model_scores(design([w for w, _ in cal_rows]), dict(method=method, **learned)),
            contextual, None if independent_application else limits)
    payload = dict(format=1, profile=PROFILE, binding=manifest["binding"], feature_set_version=version,
                   features=features, policy_version=POLICY, contextual=contextual, method=method,
                   preprocessing=params, thresholds=thresholds, application_limits=limits,
                   seed=seed, dataset_manifest_sha256=hashlib.sha256(raw).hexdigest(), **learned)
    if separate_health:
        payload["separate_health"] = True
    if incident_calibration:
        payload.update(calibration_policy=INCIDENT_CALIBRATION_POLICY, operation_context=operation_context,
                       unsupported_regimes=calibration["unsupported_regimes"])
    report = dict(binding=manifest["binding"], method=method, contextual=contextual, thresholds=thresholds,
                  threshold_mode=mode)
    if incident_calibration:
        report.update(operation_context=operation_context, incident_calibration=calibration)
        if operation_context:
            report.update(network_training_windows=len(parts["train"]), network_calibration_windows=len(parts["calibration"]),
                          http_normal_calibration_windows=len(http_normal_cal))
    else:
        report["pair_calibration"] = calibration
    report.update(separate_application_health=separate_health, independent_application_evidence=independent_application,
                  **fit_details, application_limits=limits, normal_windows=len(normal_cal),
                  normal_windows_by_schedule=dict(Counter(states)))
    if not incident_calibration:
        false_windows = sum(score > thresholds[state] or degradation(w, limits)
                            for score, state, w in zip(scores, states, normal_cal, strict=True))
        report.update(false_window_budget_per_normal_hour=BUDGET, calibration_false_windows=int(false_windows))
    return publish_bundle(destination, payload, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--network-only", action="store_true")
    parser.add_argument("--rules", action="store_true")
    parser.add_argument("--projector", type=Path, help="incident replay binary, default netwatch-projection-driver in PATH")
    args = parser.parse_args()
    print(json.dumps(fit(args.manifest, args.output, args.method, contextual=not args.network_only, rules=args.rules,
                         projector=args.projector or "netwatch-projection-driver"), indent=2))


if __name__ == "__main__":
    main()

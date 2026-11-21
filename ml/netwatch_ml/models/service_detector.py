"""Portable, service-scoped detector with optional independent HTTP evidence."""

import math

import numpy as np

from netwatch_ml.models.application_operations import (
    OPERATION_FEATURES, operation_features, operation_measurement_available,
)
from netwatch_ml.models.detector import fit_transform_parameters, forest_score, transform, window_unavailability

PROFILE = "calibrated-tcp-service-5s"
INCIDENT_CALIBRATION_POLICY = "v4"
APP_FEATURES = ("completed_requests", "server_error_fraction", "mean_latency_ms", "planned_activity")


def application_values(context):
    requests = context["completed_requests"]
    return [requests, context["server_errors"] / requests if requests else None,
            context["mean_latency_ms"], context["planned_activity"]]


def application_available(window):
    return (window.get("application") or {}).get("planned_activity") is not None


def feature_rows(windows, features, contextual, separate_health=False, use_schedule=True, operation_context=False):
    return np.asarray([
        [w["features"][f] for f in features]
        + detector_context(w, contextual, separate_health, use_schedule, operation_context)
        for w in windows
    ], dtype=float)


def detector_context(window, contextual, separate_health, use_schedule=True, operation_context=False):
    if not contextual:
        return []
    values = application_values(window["application"])
    if separate_health:
        values = [values[0], values[3]]
    if not use_schedule:
        values.pop()
    if operation_context:
        values.extend(operation_features(window["application"]))
    return values


def context_width(payload):
    if not payload["contextual"]:
        return 0
    width = 2 if payload.get("separate_health", False) else len(APP_FEATURES)
    if payload.get("calibration_policy") == INCIDENT_CALIBRATION_POLICY:
        width -= 1
    if payload.get("operation_context", False):
        width += len(OPERATION_FEATURES)
    return width


def planned_state(window, contextual):
    return "planned" if contextual and window["application"]["planned_activity"] else "unplanned"


def train_scaler(windows, features, contextual, separate_health=False, use_schedule=True, operation_context=False):
    return fit_transform_parameters(feature_rows(windows, features, contextual, separate_health,
                                                use_schedule, operation_context))


def model_scores(values, payload):
    if payload["method"] == "isolation_forest":
        return np.asarray([forest_score(row, payload["forest"]) for row in values])
    hidden, output = payload["layers"]
    values = np.maximum(values @ np.asarray(hidden[0]) + np.asarray(hidden[1]), 0)
    logits = (values @ np.asarray(output[0]) + np.asarray(output[1])).ravel()
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def degradation(window, limits):
    return http_evidence(window, limits)["status"] == "anomalous"


def http_evidence(window, limits):
    context = window.get("application")
    if limits is None:
        return {"status": "unavailable", "reason": "unconfigured"}
    if context is None:
        return {"status": "unavailable", "reason": "missing_telemetry"}
    if context["completed_requests"] < limits["min_requests"]:
        return {"status": "unavailable", "reason": "insufficient_requests"}
    _, error_fraction, latency, _ = application_values(context)
    errors = error_fraction >= limits["server_error_fraction"]
    slow = latency >= limits["latency_ms"]
    if errors or slow:
        reason = "errors_and_latency" if errors and slow else ("error_fraction" if errors else "latency")
        return {"status": "anomalous", "reason": reason}
    return {"status": "normal", "reason": "within_limits"}


def validate_config(payload, width):
    contextual = payload["contextual"]
    separate_health = payload.get("separate_health", False)
    if type(contextual) is not bool or type(separate_health) is not bool or (separate_health and not contextual):
        raise ValueError("invalid context configuration")
    policy = payload.get("calibration_policy")
    operations = payload.get("operation_context", False)
    if (policy not in (None, INCIDENT_CALIBRATION_POLICY) or type(operations) is not bool
            or (operations and (not contextual or policy is None))):
        raise ValueError("invalid incident calibration configuration")
    method = payload["method"]
    if method not in ("isolation_forest", "mlp") or ("forest" in payload) != (method == "isolation_forest"):
        raise ValueError("invalid detector method")
    thresholds = payload["thresholds"]
    if set(thresholds) != ({"planned", "unplanned"} if contextual else {"unplanned"}):
        raise ValueError("invalid service thresholds")
    if policy is None:
        if "unsupported_regimes" in payload or any(type(t) not in (int, float) or not 0 <= t <= 1
                                                   for t in thresholds.values()):
            raise ValueError("invalid legacy service thresholds")
    else:
        if (all(t is None for t in thresholds.values())
                or any(t is not None and (type(t) not in (int, float) or not 0 <= t < 1) for t in thresholds.values())
                or payload.get("unsupported_regimes") != sorted(name for name, t in thresholds.items() if t is None)):
            raise ValueError("invalid incident calibration thresholds")
    limits = payload["application_limits"]
    if limits is not None:
        if not contextual:
            raise ValueError("application rules require context")
        validate_application_limits(limits)
    if method == "mlp":
        hidden, output = payload["layers"]
        weights, bias = map(np.asarray, hidden)
        if bias.ndim != 1 or not 1 <= bias.size <= 64 or weights.shape != (width * 2, bias.size):
            raise ValueError("invalid MLP hidden layer")
        out_weights, out_bias = map(np.asarray, output)
        if out_weights.shape != (bias.size, 1) or out_bias.shape != (1,) or not all(
            np.isfinite(array).all() for array in (weights, bias, out_weights, out_bias)
        ):
            raise ValueError("invalid MLP output layer")


def validate_application_limits(limits):
    if (set(limits) != {"min_requests", "server_error_fraction", "latency_ms"}
            or type(limits["min_requests"]) is not int or limits["min_requests"] < 1
            or not 0 < limits["server_error_fraction"] <= 1 or not 0 < limits["latency_ms"] < math.inf):
        raise ValueError("invalid application limits")


def predict(window, bundle, *, independent_evidence=False):
    payload = bundle["detector"]
    result = dict(schema_version=1, event_id=window["event_id"], model_id=bundle["model_id"],
                  feature_set_version=window["feature_set_version"], status="insufficient_data",
                  score=None, threshold=None)
    reason = window_unavailability(window, payload["binding"], payload["feature_set_version"])
    if independent_evidence and reason not in ("outside_model_scope", "feature_set_mismatch"):
        result["http"] = http_evidence(window, payload["application_limits"])
    if reason:
        return dict(result, unavailable_reason=reason)
    unhealthy = not independent_evidence and degradation(window, payload["application_limits"])
    if payload["contextual"] and not application_available(window):
        if not unhealthy:
            return dict(result, unavailable_reason="application_context_missing")
        # HTTP failure evidence is independent of the availability of the calendar.
        threshold = max((t for t in payload["thresholds"].values() if t is not None), default=0.)
        return dict(result, score=math.nextafter(threshold, math.inf), threshold=threshold, status="anomalous")
    threshold = payload["thresholds"][planned_state(window, payload["contextual"])]
    if threshold is None or (independent_evidence and threshold == 1.):
        return dict(result, unavailable_reason="regime_uncalibrated")
    operation_context = payload.get("operation_context", False)
    if operation_context and not operation_measurement_available(window["application"]):
        return dict(result, unavailable_reason="application_context_missing")
    raw = feature_rows([window], payload["features"], payload["contextual"], payload.get("separate_health", False),
                       payload.get("calibration_policy") != INCIDENT_CALIBRATION_POLICY, operation_context)
    score = float(model_scores(transform(raw, payload["preprocessing"]), payload)[0])
    if unhealthy:
        score = max(score, math.nextafter(threshold, math.inf))
    return dict(result, score=score, threshold=threshold, status="anomalous" if score > threshold else "normal")

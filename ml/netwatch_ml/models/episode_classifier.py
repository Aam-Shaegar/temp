"""Portable multiclass scoring for completed TCP service episodes."""

import hashlib
import hmac
import json

import numpy as np

from netwatch_ml.models.detector import FEATURES, FEATURE_SETS, LIFECYCLE_VERSION, VERSION, matrix, transform, validate_binding
from netwatch_ml.models.application_operations import OPERATION_FEATURES, operation_features
from netwatch_ml.models.service_detector import (
    APP_FEATURES, application_available, application_values, degradation, feature_rows, validate_application_limits,
)


NORMAL = "normal"
UNKNOWN = "unknown"
LABELS = (
    NORMAL,
    "foreground_pause",
    "large_upload",
    "large_response",
    "connection_surge",
    "persistent_exchange",
)
WINDOW_NS = 5_000_000_000
MINIMUM_WINDOWS = 2
REQUEST_RATIOS = ("in_bytes_per_completed_request", "out_bytes_per_completed_request", "tcp_attempts_per_completed_request")
BASE_STATISTICS = ("median", "iqr")
LIFECYCLE_STATISTICS = (*BASE_STATISTICS, "rise", "fall")
OPERATION_EPISODE_VERSION = 5
FOREST_VERSIONS = (3, 4, OPERATION_EPISODE_VERSION)
ROUTE_FEATURES = ("request_body_excess_on_route", "response_body_excess_on_route")
OPERATION_EPISODE_FEATURES = (*OPERATION_FEATURES, *ROUTE_FEATURES)
MAX_OPERATION_REFERENCES = 512
BODY_FIELDS = ("request_body_bytes", "response_body_bytes")


def episode_feature_names(feature_set_version, contextual, request_ratios, operation_names=()):
    features = list(FEATURE_SETS[feature_set_version])
    if contextual:
        features.extend(APP_FEATURES)
    if request_ratios:
        features.extend(REQUEST_RATIOS)
    features.extend(operation_names)
    return features


def route_key(operation):
    return f"{operation['method']} {operation['route_id']}"


def body_per_request(operation, field):
    size, count = operation[field], operation["completed_requests"]
    return size / count if size is not None and count else None


def route_excess(application, reference):
    excess = [[], []]
    for operation in application.get("operations", ()):
        limits = reference.get(route_key(operation))
        if limits is None:
            continue
        for values, field, limit in zip(excess, BODY_FIELDS, limits, strict=True):
            size = body_per_request(operation, field)
            if size is not None and limit is not None:
                values.append(max(0., size / max(1., limit) - 1.))
    return [max(values, default=None) for values in excess]


def operation_episode_values(application, reference):
    return list(operation_features(application)) + route_excess(application, reference)


def quantiles(raw, levels):
    values = np.full((len(levels), raw.shape[1]), np.nan)
    available = np.isfinite(raw).any(axis=0)
    if available.any():
        values[:, available] = np.nanquantile(raw[:, available], levels, axis=0)
    return values


def episode_features(windows, contextual=False, request_ratios=False, feature_set_version=VERSION,
                     operation_reference=None, operation_names=OPERATION_EPISODE_FEATURES):
    """Return robust summaries of consecutive, fully observed service windows."""
    features = FEATURE_SETS.get(feature_set_version)
    if features is None or len(windows) < MINIMUM_WINDOWS:
        return None
    required = set(features)
    identity = tuple(windows[0].get(key) for key in ("sensor_id", "entity_id", "capture_epoch"))
    previous = None
    for window in windows:
        start, end = int(window["window_start_ns"]), int(window["window_end_ns"])
        quality = window["quality"]
        if (
            window["entity_type"] != "destination_service"
            or window["feature_set_version"] != feature_set_version
            or not required.issubset(window["features"])
            or tuple(window.get(key) for key in ("sensor_id", "entity_id", "capture_epoch")) != identity
            or start % WINDOW_NS != 0
            or end - start != WINDOW_NS
            or (previous is not None and start != previous)
            or quality["coverage"] != "complete"
            or any(quality[key] != 0 for key in ("capture_dropped", "late_packets", "state_evictions"))
        ):
            return None
        previous = end
    if operation_reference is None and (request_ratios or contextual) and not all(application_available(w) for w in windows):
        return None
    if operation_reference is None:
        raw = feature_rows(windows, features, contextual) if contextual else matrix([w["features"] for w in windows], features)
    else:
        packets = matrix([w["features"] for w in windows], features)
        context = [application_values(w["application"]) if w.get("application") else [None] * len(APP_FEATURES)
                   for w in windows]
        raw = np.column_stack((packets, np.asarray(context, dtype=float))) if contextual else packets
    if request_ratios:
        ratios = []
        for window in windows:
            count = (window.get("application") or {}).get("completed_requests")
            values = window["features"]
            numerators = (values["in_ip_bytes"], values["out_ip_bytes"], values["new_tcp_attempts"])
            ratios.append([value / count if count and value is not None else np.nan for value in numerators])
        raw = np.column_stack((raw, ratios))
    if operation_reference is not None and operation_names:
        applications = [w.get("application") or {} for w in windows]
        operations = [operation_episode_values(app, operation_reference) for app in applications]
        positions = [OPERATION_EPISODE_FEATURES.index(name) for name in operation_names]
        raw = np.column_stack((raw, np.asarray(operations, dtype=float)[:, positions]))
    if np.isinf(raw).any() or (raw < 0).any():
        return None
    low, mid, high = quantiles(raw, (0.25, 0.5, 0.75))
    summary = [mid, high - low]
    if feature_set_version == LIFECYCLE_VERSION:
        half = len(raw) // 2
        early, = quantiles(raw[:half], (0.5,))
        late, = quantiles(raw[-half:], (0.5,))
        delta = late - early
        # The portable transform accepts nonnegative values. Preserve direction
        # and missing observations instead of replacing the latter with zero.
        summary.extend((np.maximum(delta, 0), np.maximum(-delta, 0)))
    return np.concatenate(summary)


def softmax(logits, temperature):
    scores = np.asarray(logits, dtype=float)
    scores = (scores - np.max(scores, axis=-1, keepdims=True)) / temperature
    exp = np.exp(scores)
    return exp / exp.sum(axis=-1, keepdims=True)


def probabilities(features, bundle):
    values = transform(np.atleast_2d(features), bundle["preprocessing"])
    spec = bundle["model"]
    if bundle["method"] == "logistic":
        logits = values @ np.asarray(spec["coefficients"]).T + spec["intercepts"]
    elif bundle["method"] == "mlp":
        for weights, biases in spec["layers"][:-1]:
            values = np.maximum(values @ weights + biases, 0)
        weights, biases = spec["layers"][-1]
        logits = values @ weights + biases
    elif bundle["method"] == "random_forest":
        return softmax(np.log(np.maximum(forest_probabilities(values, spec), np.finfo(float).tiny)), bundle["temperature"])
    elif bundle["method"] == "histogram":
        logits = np.broadcast_to(spec["intercepts"], (len(values), len(LABELS))).copy()
        for stage in spec["stages"]:
            for label, tree in enumerate(stage):
                for i, row in enumerate(values):
                    feature, threshold, left, right, value = tree[0]
                    while left != -1:
                        feature, threshold, left, right, value = tree[left if row[feature] <= threshold else right]
                    logits[i, label] += value
    else:
        raise ValueError("unknown episode classifier")
    return softmax(logits, bundle["temperature"])


def forest_probabilities(values, spec):
    # sklearn trees compare float32 features with float64 thresholds.
    values = np.asarray(values, dtype=np.float32)
    probabilities = np.zeros((len(values), len(LABELS)))
    for tree in spec["trees"]:
        for i, row in enumerate(values):
            index = 0
            while tree[index][2] != -1:
                feature, threshold, left, right, _ = tree[index]
                index = left if float(row[feature]) <= threshold else right
            probabilities[i] += tree[index][4]
    return probabilities / len(spec["trees"])


def validate_random_forest(bundle):
    trees = bundle["model"]["trees"]
    width = len(bundle["features"]) * len(bundle["statistics"])
    params = bundle["preprocessing"]
    center, scale = np.asarray(params["median"], dtype=float), np.asarray(params["scale"], dtype=float)
    thresholds = bundle["class_thresholds"]
    if (center.shape != (width,) or scale.shape != (width,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any()
            or set(thresholds) != set(LABELS)
            or not all(0 <= value <= np.nextafter(1., 2.) for value in thresholds.values())
            or not 1 <= len(trees) <= 256):
        raise ValueError("invalid episode forest")
    for tree in trees:
        if not 1 <= len(tree) <= 8191:
            raise ValueError("invalid episode tree size")
        for i, (feature, threshold, left, right, value) in enumerate(tree):
            if type(left) is not int or type(right) is not int:
                raise ValueError("invalid episode tree links")
            if left == right == -1:
                p = np.asarray(value, dtype=float)
                if p.shape != (len(LABELS),) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1.):
                    raise ValueError("invalid leaf probabilities")
            elif (type(feature) is not int or not 0 <= feature < width * 2
                  or not np.isfinite(threshold) or not i < left < len(tree) or not i < right < len(tree)):
                raise ValueError("invalid episode tree split")


def validate_operation_reference(reference):
    if not isinstance(reference, dict) or len(reference) > MAX_OPERATION_REFERENCES:
        raise ValueError("invalid route reference")
    for key, values in reference.items():
        if (not isinstance(key, str) or not 3 <= len(key) <= 160 or key.count(" ") != 1
                or not isinstance(values, list) or len(values) != len(BODY_FIELDS)
                or all(value is None for value in values)
                or any(type(value) not in (int, float) or not np.isfinite(value) or value < 0
                       for value in values if value is not None)):
            raise ValueError("invalid route reference")


def load(path, expected_sha256, *, allow_experimental=False):
    if path.stat().st_size > 32 << 20:
        raise ValueError("episode classifier bundle too large")
    raw = path.read_bytes()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        raise ValueError("episode classifier checksum mismatch")
    bundle = json.loads(raw)
    version = bundle["schema_version"]
    forest = version in FOREST_VERSIONS
    operations = version == OPERATION_EPISODE_VERSION
    lifecycle = version in (4, OPERATION_EPISODE_VERSION)
    feature_set = LIFECYCLE_VERSION if lifecycle else VERSION
    operation_names = bundle["operation_features"] if operations else []
    if (not isinstance(operation_names, list) or any(not isinstance(name, str) for name in operation_names)
            or operation_names != [name for name in OPERATION_EPISODE_FEATURES if name in operation_names]):
        raise ValueError("invalid episode operation features")
    features = episode_feature_names(feature_set, contextual=version >= 2, request_ratios=forest,
                                    operation_names=operation_names)
    statistics = LIFECYCLE_STATISTICS if lifecycle else BASE_STATISTICS
    methods = ("random_forest",) if forest else ("logistic", "histogram", "mlp")
    if (version not in (1, 2, *FOREST_VERSIONS) or bundle["labels"] != list(LABELS) or bundle["features"] != features
            or bundle.get("feature_set_version", VERSION) != feature_set
            or bundle["statistics"] != list(statistics) or bundle["method"] not in methods
            or not 0.25 <= bundle["temperature"] <= 4.0
            or not forest and not 0 <= bundle["confidence_threshold"] <= np.nextafter(1.0, 2.0)):
        raise ValueError("unsupported episode classifier")
    if version >= 2:
        validate_binding(bundle["binding"])
        validate_application_limits(bundle["application_limits"])
    if forest:
        validate_random_forest(bundle)
    if operations:
        validate_operation_reference(bundle["operation_reference"])
        excluded = bundle.get("experimental_held_out_type")
        if excluded is not None and (not allow_experimental or excluded not in LABELS[1:]
                                     or bundle["class_thresholds"][excluded] <= 1):
            raise ValueError("experimental leave-one-type-out model")
    return bundle


def application_degraded(windows, limits):
    evidence = [degradation(window, limits) for window in windows]
    return any(a and b for a, b in zip(evidence, evidence[1:]))


def journal_windows(journal):
    """Reconstruct the confirmed span from one operator journal response."""
    incident = journal["incident"]
    end = incident["estimated_end_ns"]
    if end is None:
        return []
    start, end = int(incident["first_ns"]), int(end)
    windows = []
    for event in journal["events"]:
        if start <= int(event["start_ns"]) < int(event["end_ns"]) <= end:
            window = dict(
                schema_version=1, event_id=event["event"], sensor_id=incident["sensor"],
                feature_set_version=journal["feature_set_version"], capture_epoch=incident["epoch"],
                entity_type=incident["entity_type"], entity_id=incident["entity"],
                window_start_ns=event["start_ns"], window_end_ns=event["end_ns"],
                features=event["features"], quality=event["quality"],
            )
            if event.get("application") is not None:
                window["application"] = event["application"]
            windows.append(window)
    if not windows or int(windows[0]["window_start_ns"]) != start or int(windows[-1]["window_end_ns"]) != end:
        return []
    return windows


def merge_journal_pages(pages):
    """Join ordered operator API pages without crossing incident identities."""
    if not isinstance(pages, list) or not pages or not all(isinstance(page, dict) for page in pages):
        raise ValueError("expected incident journal pages")
    first = pages[0]
    identity_keys = ("id", "sensor", "epoch", "entity", "entity_type", "model", "policy",
                     "evidence_channel", "first_ns", "estimated_end_ns")
    incident = first["incident"]
    identity = tuple(incident.get(key) for key in identity_keys)
    if any(incident.get(key) is None for key in ("id", "sensor", "epoch", "first_ns")):
        raise ValueError("journal pages lack an incident identity")
    events = []
    previous = None
    for index, page in enumerate(pages):
        if (tuple(page["incident"].get(key) for key in identity_keys) != identity
                or page["feature_set_version"] != first["feature_set_version"]
                or page.get("rule") != first.get("rule")
                or (index + 1 < len(pages) and page.get("more") is not True)):
            raise ValueError("journal pages describe different incident versions")
        for event in page["events"]:
            start = int(event["start_ns"])
            if previous is not None and start <= previous:
                raise ValueError("journal pages are out of order or repeat a window")
            events.append(event)
            previous = start
    return dict(first, events=events, more=pages[-1].get("more", False))


def classify(windows, bundle, *, evidence_channel="combined"):
    if evidence_channel not in ("combined", "network", "http"):
        raise ValueError("unsupported episode evidence channel")
    version = bundle.get("schema_version")
    forest = version in FOREST_VERSIONS
    contextual = forest or version == 2
    operations = version == OPERATION_EPISODE_VERSION
    feature_set = LIFECYCLE_VERSION if version in (4, OPERATION_EPISODE_VERSION) else VERSION
    if contextual and any(not all(w.get(key) == expected for key, expected in bundle["binding"].items()) for w in windows):
        return {"label": UNKNOWN, "reason": "outside_model_scope"}
    if evidence_channel == "http":
        if contextual and any(
            a.get("capture_epoch") == b.get("capture_epoch")
            and a.get("feature_set_version") == b.get("feature_set_version") == feature_set
            and int(a["window_end_ns"]) == int(b["window_start_ns"])
            and degradation(a, bundle["application_limits"]) and degradation(b, bundle["application_limits"])
            for a, b in zip(windows, windows[1:])
        ):
            return {"label": UNKNOWN, "reason": "application_degradation"}
        return {"label": UNKNOWN, "reason": "insufficient_data"}
    features = episode_features(windows, contextual, request_ratios=forest, feature_set_version=feature_set,
                                operation_reference=bundle["operation_reference"] if operations else None,
                                operation_names=bundle["operation_features"] if operations else ())
    if features is None:
        return {"label": UNKNOWN, "reason": "insufficient_data"}
    if contextual and evidence_channel == "combined" and application_degraded(windows, bundle["application_limits"]):
        return {"label": UNKNOWN, "reason": "application_degradation"}
    scores = probabilities(features, bundle)[0]
    index = int(np.argmax(scores))
    threshold = bundle["class_thresholds"][LABELS[index]] if forest else bundle["confidence_threshold"]
    if scores[index] < threshold:
        reason = "uncalibrated_class" if operations and threshold > 1 else "low_confidence"
        return {"label": UNKNOWN, "reason": reason}
    if operations and LABELS[index] == NORMAL and evidence_channel == "network":
        return {"label": NORMAL, "reason": "normal_like_network_incident"}
    # Calibration on a handful of labelled runs does not support an operator-facing probability.
    return {"label": LABELS[index]}

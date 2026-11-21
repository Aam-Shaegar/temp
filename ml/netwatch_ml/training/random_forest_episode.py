"""Fit one portable Random Forest shared by each service's detector candidates."""

from collections import Counter, defaultdict
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from netwatch_ml.models.detector import LIFECYCLE_VERSION, fit_transform_parameters, transform
from netwatch_ml.models.episode_classifier import (
    BASE_STATISTICS, BODY_FIELDS, LABELS, LIFECYCLE_STATISTICS, NORMAL, OPERATION_EPISODE_FEATURES,
    OPERATION_EPISODE_VERSION, UNKNOWN, application_degraded, body_per_request, classify, episode_feature_names,
    episode_features, forest_probabilities, load, operation_episode_values, probabilities, route_key,
    validate_operation_reference,
)
from netwatch_ml.training.baselines import sha
from netwatch_ml.training.episode_classification import fit_temperature, write_json
from netwatch_ml.training.service_detector import application_limits, load_partitions


def expand_probabilities(values, classes):
    expanded = np.zeros((len(values), len(LABELS)))
    expanded[:, classes] = values
    return expanded


def encode_forest(model):
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        probabilities = expand_probabilities(tree.value[:, 0, :], model.classes_)
        trees.append([
            [int(f), float(t), int(left), int(right), value.tolist() if left == -1 else None]
            for f, t, left, right, value in zip(
                tree.feature, tree.threshold, tree.children_left, tree.children_right, probabilities, strict=True
            )
        ])
    return {"trees": trees}


def edge_samples(windows):
    yield windows
    if len(windows) >= 3:
        yield windows[1:]
        yield windows[:-1]
    if len(windows) >= 4:
        yield windows[1:-1]


def class_thresholds(scores, truth, degraded):
    """Set type-specific acceptance on development episodes."""
    winners = scores.argmax(axis=1)
    result = {NORMAL: 0.0}
    for index, label in enumerate(LABELS[1:], start=1):
        confidence = scores[~degraded & (winners == index), index]
        correct = truth[~degraded & (winners == index)] == index
        if not np.any(correct):
            result[label] = float(np.nextafter(1.0, 2.0))
            continue
        # Any higher threshold up to the first accepted true case would give the
        # same calibration decisions and reject more of the unobserved range.
        result[label] = float(np.nextafter(max(confidence[~correct], default=0.0), np.inf))
    return result


def fit_operation_reference(training):
    """Learn route body-size references from normal windows in separate captures."""
    samples = defaultdict(list)
    for window, label in training:
        if label != NORMAL:
            continue
        for operation in (window.get("application") or {}).get("operations", ()):
            for field in BODY_FIELDS:
                size = body_per_request(operation, field)
                if size is not None:
                    samples[route_key(operation), field].append((window["capture_epoch"], size))

    reference = {}
    for key in sorted({key for key, _ in samples}):
        values = []
        for field in BODY_FIELDS:
            measured = samples[key, field]
            supported = len(measured) >= 3 and len({capture for capture, _ in measured}) >= 2
            values.append(float(np.quantile([size for _, size in measured], .9)) if supported else None)
        if any(value is not None for value in values):
            reference[key] = values
    validate_operation_reference(reference)
    return reference


def supported_operations(training, reference, leave_out_type):
    """Keep varying features with normal measurements in distinct training captures."""
    groups = defaultdict(set)
    values_seen = defaultdict(set)
    for window, label in training:
        # The partition loader retains the binary label only. In a held-type
        # experiment rely on ordinary measurements to avoid touching any of the
        # omitted anomalous windows during feature selection.
        if leave_out_type is not None and label != NORMAL:
            continue
        values = operation_episode_values(window.get("application") or {}, reference)
        for name, value in zip(OPERATION_EPISODE_FEATURES, values, strict=True):
            if value is None:
                continue
            values_seen[name].add(value)
            if label == NORMAL:
                groups[name].add(window["capture_epoch"])
    return [name for name in OPERATION_EPISODE_FEATURES if len(groups[name]) >= 2 and len(values_seen[name]) >= 2]


def supported_thresholds(scores, truth, sources, eligible):
    """Require an observed error-free operating point supported by multiple captures."""
    winners = scores.argmax(axis=1)
    result, details = {NORMAL: 0.}, {}
    for index, label in enumerate(LABELS[1:], start=1):
        positive = eligible & (truth == index)
        candidate = eligible & (winners == index)
        correct = candidate & positive
        floor = float(np.nextafter(max(scores[candidate & ~positive, index], default=0.), np.inf))
        possible = correct & (scores[:, index] >= floor)
        supported = len(set(sources[possible])) >= 2
        if supported:
            status = "calibrated"
        elif not positive.any():
            status = "missing_calibration_cases"
        elif not correct.any():
            status = "no_correct_winners"
        elif not possible.any():
            status = "no_error_free_cutoff"
        else:
            status = "insufficient_capture_support"
        result[label] = floor if supported else float(np.nextafter(1., 2.))
        accepted = candidate & (scores[:, index] >= result[label])
        details[label] = dict(
            positive=int(positive.sum()), positive_captures=len(set(sources[positive])),
            correct_winners=int(correct.sum()), accepted=int(accepted.sum()),
            accepted_captures=len(set(sources[accepted])),
            precision=float((accepted & positive).sum() / accepted.sum()) if accepted.any() else None,
            coverage=float((accepted & positive).sum() / positive.sum()) if positive.any() else None,
            status=status,
        )
    return result, details


def rejected_unknown_cases(cases, reasons):
    return dict(
        episodes=len(cases), captures=len({case["source"] for case, _ in cases}),
        rejected=sum(outcome["label"] == UNKNOWN and outcome.get("reason") in reasons for _, outcome in cases),
        abstained=sum(outcome["label"] == UNKNOWN for _, outcome in cases),
        outcomes=Counter(outcome.get("reason", outcome["label"]) for _, outcome in cases),
    )


def fit(manifest_path, output, seed=42, *, leave_out_type=None):
    from netwatch_ml.training.contextual_classification import episodes

    manifest = json.loads(manifest_path.read_text())
    version = manifest["feature_set_version"]
    episode_version = manifest.get("episode_classifier_version", 4 if version == LIFECYCLE_VERSION else 3)
    operations = episode_version == OPERATION_EPISODE_VERSION
    if (type(episode_version) is not int
            or (version == LIFECYCLE_VERSION and episode_version not in (4, OPERATION_EPISODE_VERSION))
            or (version != LIFECYCLE_VERSION and episode_version != 3)
            or (leave_out_type is not None and (not operations or leave_out_type not in LABELS[1:]))):
        raise ValueError("unsupported episode classifier version or hold-out type")
    parts = load_partitions(manifest, manifest_path.parent, contextual=not operations)
    normal_cal = [w for w, label in parts["calibration"] if label == NORMAL and (not operations or w.get("application"))]
    limits = application_limits(normal_cal, manifest.get("application_limits"))
    reference = fit_operation_reference(parts["train"]) if operations else None
    operation_names = supported_operations(parts["train"], reference, leave_out_type) if operations else []
    provenance = [] if operations else None
    projected_train, projected_calibration = Counter(), Counter()
    short_train, short_calibration = Counter(), Counter()
    training = [
        row for row in episodes(manifest, manifest_path.parent, "train",
                                request_ratios=True, projection_counts=projected_train if leave_out_type is None else None,
                                short_counts=short_train,
                                operation_reference=reference, operation_names=operation_names)
        if row[0] not in (UNKNOWN, leave_out_type)
    ]
    calibration = episodes(manifest, manifest_path.parent, "calibration",
                           request_ratios=True,
                           projection_counts=projected_calibration if leave_out_type is None else None,
                           short_counts=short_calibration,
                           operation_reference=reference, operation_names=operation_names, provenance=provenance)
    if leave_out_type is not None:
        kept = [(row, meta) for row, meta in zip(calibration, provenance, strict=True) if row[0] != leave_out_type]
        calibration = [row for row, _ in kept]
        provenance = [meta for _, meta in kept]
    if {row[0] for row in training} != set(LABELS) - {leave_out_type}:
        raise ValueError("train needs every configured episode class")
    if not any(row[0] != UNKNOWN for row in calibration):
        raise ValueError("calibration needs labeled known observations")
    if not operations and {row[0] for row in calibration} != set((*LABELS, UNKNOWN)):
        raise ValueError("legacy episode calibration needs all known classes and unknowns")

    def features(windows):
        return episode_features(windows, contextual=True, request_ratios=True,
                                feature_set_version=version, operation_reference=reference,
                                operation_names=operation_names)

    pre = fit_transform_parameters(np.asarray([row[2] for row in training]))
    class_size = Counter(label for label, _, _ in training)
    x, y, weights = [], [], []
    for label, windows, _ in training:
        variants = list(edge_samples(windows))
        for variant in variants:
            x.append(features(variant))
            y.append(LABELS.index(label))
            weights.append(1 / (len(variants) * class_size[label]))
    model = RandomForestClassifier(n_estimators=96, max_depth=8, max_features="sqrt",
                                   min_samples_leaf=1, random_state=seed, n_jobs=1)
    model.fit(transform(np.asarray(x), pre), y, sample_weight=weights)
    raw_cal = np.asarray([row[2] for row in calibration])
    cal_x = transform(raw_cal, pre)
    truth = np.asarray([LABELS.index(label) if label != UNKNOWN else len(LABELS) for label, _, _ in calibration])
    known = truth < len(LABELS)
    predicted = expand_probabilities(model.predict_proba(cal_x), model.classes_)
    logits = np.log(np.maximum(predicted, np.finfo(float).tiny))
    temperature = fit_temperature(logits[known], truth[known])
    spec = encode_forest(model)
    np.testing.assert_allclose(forest_probabilities(cal_x, spec), predicted, rtol=0, atol=1e-12)
    lifecycle = version == LIFECYCLE_VERSION
    statistics = LIFECYCLE_STATISTICS if lifecycle else BASE_STATISTICS
    bundle = dict(schema_version=episode_version, method="random_forest", labels=LABELS,
                  features=episode_feature_names(version, contextual=True, request_ratios=True,
                                                operation_names=operation_names),
                  statistics=list(statistics),
                  binding=manifest["binding"], preprocessing=pre, model=spec, temperature=temperature, application_limits=limits)
    if lifecycle:
        bundle["feature_set_version"] = version
    scores = probabilities(raw_cal, bundle)
    degraded = np.asarray([application_degraded(w, limits) for _, w, _ in calibration])
    if operations:
        bundle["operation_reference"] = reference
        bundle["operation_features"] = operation_names
        if leave_out_type is not None:
            bundle["experimental_held_out_type"] = leave_out_type
        http_only = np.asarray([label == UNKNOWN and meta["channel"] != "network"
                                and (meta["channel"] == "http" or bad_http)
                                for (label, _, _), meta, bad_http in zip(calibration, provenance, degraded, strict=True)])
        sources = np.asarray([meta["source"] for meta in provenance])
        bundle["class_thresholds"], support = supported_thresholds(scores, truth, sources, ~http_only)
    else:
        bundle["class_thresholds"] = class_thresholds(scores, truth, degraded)
    output.mkdir(parents=True)
    path = output / "random_forest.json"
    write_json(path, bundle)
    digest = sha(path)
    checked = load(path, digest, allow_experimental=leave_out_type is not None)
    report = dict(
        sha256=digest, train_episodes=len(training), train_samples=len(x), cal_episodes=len(calibration),
        projected_train_episodes=sum(count for label, count in projected_train.items() if label != UNKNOWN),
        projected_cal_episodes=sum(projected_calibration.values()),
        short_oracle_train=dict(short_train), short_oracle_calibration=dict(short_calibration),
        calibration=Counter(f"{label}:{classify(windows, checked)['label']}" for label, windows, _ in calibration),
    )
    if operations:
        outcomes = [classify(windows, checked, evidence_channel="http" if http else "network")
                    for (_, windows, _), http in zip(calibration, http_only, strict=True)]
        unknown_cases = [(meta, outcome, http)
                         for (label, _, _), meta, outcome, http in zip(calibration, provenance, outcomes, http_only, strict=True)
                         if label == UNKNOWN]
        network_unknown = [(meta, outcome) for meta, outcome, _ in unknown_cases if meta["channel"] == "network"]
        http_unknown = [(meta, outcome) for meta, outcome, http in unknown_cases if http]
        report.update(
            calibration_scope="development_only", class_support=support, operation_reference_routes=len(reference),
            operation_features=operation_names, experimental_held_out_type=leave_out_type,
            network_open_set_calibration=(rejected_unknown_cases(network_unknown, {"low_confidence", "unknown_type"})
                                          if network_unknown else None),
            http_unknown_calibration=rejected_unknown_cases(http_unknown, {"application_degradation"}),
            unknown_without_network_evidence=len(unknown_cases) - len(network_unknown) - len(http_unknown),
            calibration=Counter(f"{label}:{outcome['label']}"
                                for (label, _, _), outcome in zip(calibration, outcomes, strict=True)),
        )
    write_json(output / "manifest.json", dict(dataset_manifest_sha256=sha(manifest_path), models={"random_forest": report}))
    return report

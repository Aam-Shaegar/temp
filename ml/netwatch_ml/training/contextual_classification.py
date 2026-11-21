"""Fit logistic regression and gradient boosting for one service's completed episodes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from netwatch_ml.models.detector import FEATURES, VERSION, fit_transform_parameters, transform
from netwatch_ml.models.episode_classifier import (
    LABELS, MINIMUM_WINDOWS, NORMAL, OPERATION_EPISODE_FEATURES, UNKNOWN, application_degraded,
    classify, episode_features, probabilities,
)
from netwatch_ml.models.service_detector import APP_FEATURES
from netwatch_ml.training.baselines import sha
from netwatch_ml.training.episode_classification import (
    encode_model, fit_temperature, models, uncalibrated, write_json,
)
from netwatch_ml.training.service_detector import application_limits, load_partitions

METHODS = ("logistic", "histogram")


def capture_bytes(root, relative, digest):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("episode paths must stay within the dataset")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != digest:
        raise ValueError("episode source digest mismatch")
    return contents


def projected_samples(capture, root, rows):
    source = capture.get("projected_episodes")
    if source is None:
        return []

    sidecar = json.loads(capture_bytes(root, source["path"], source["sha256"]))
    if sidecar["source_sha256"] != capture["source_sha256"]:
        raise ValueError("projected episodes must belong to the enclosing capture")
    by_id = {row["window"]["event_id"]: row for row in rows}
    episodes = []
    for item in sidecar["episodes"]:
        detector, label, ids = item["detector"], item["episode_label"], item["event_ids"]
        if (detector not in ("isolation_forest", "mlp") or label not in (*LABELS, UNKNOWN)
                or not isinstance(ids, list) or any(not isinstance(i, str) or i not in by_id for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("invalid projected episode or cross-capture window reference")
        evidence = [by_id[i] for i in ids]
        annotated = {row["episode_label"] for row in evidence if row["phase"] is not None}
        if label == NORMAL and annotated != {NORMAL}:
            raise ValueError("normal projected episode needs independently labeled normal evidence")
        if label not in (NORMAL, UNKNOWN) and annotated - {NORMAL} != {label}:
            raise ValueError("projected episode type conflicts with independent labels")
        episodes.append((label, [row["window"] for row in evidence]))
    return episodes


def episodes(manifest, root, part, *, request_ratios=False, projection_counts=None, short_counts=None,
             operation_reference=None, operation_names=OPERATION_EPISODE_FEATURES, provenance=None):
    output = []
    version = manifest["feature_set_version"]
    for capture in manifest["captures"][part]:
        content = capture_bytes(root, capture["windows"], capture["windows_sha256"])
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        phases = {}
        for row in rows:
            if row["phase"] is not None:
                phases.setdefault(row["phase"], []).append(row)
        oracle = []
        for group in phases.values():
            group.sort(key=lambda row: int(row["window"]["window_start_ns"]))
            label = group[0]["episode_label"]
            if (label not in (*LABELS, UNKNOWN)
                    or any(row["episode_label"] != label or row["label"] != ("normal" if label == NORMAL else "anomaly")
                           for row in group)):
                raise ValueError("invalid independently labeled episode")
            if len(group) < MINIMUM_WINDOWS and short_counts is not None:
                short_counts[label] += 1
                continue
            channel = group[0].get("episode_evidence_channel") if provenance is not None else None
            if provenance is not None and (
                channel not in (None, "network", "http") or (channel == "http" and label != UNKNOWN)
                or any(row.get("episode_evidence_channel") != channel for row in group)
            ):
                raise ValueError("invalid independent episode evidence channel")
            oracle.append((label, [row["window"] for row in group], channel))

        proposals = projected_samples(capture, root, rows) if projection_counts is not None else []
        candidates = [(False, *sample) for sample in oracle] + [(True, *sample, None) for sample in proposals]
        by_span = {}
        for projected, label, windows, channel in candidates:
            features = episode_features(windows, contextual=True, request_ratios=request_ratios,
                                        feature_set_version=version, operation_reference=operation_reference,
                                        operation_names=operation_names)
            if features is None:
                raise ValueError("episode windows must be complete and chronological")
            span = tuple(w["event_id"] for w in windows)
            if span in by_span:
                if by_span[span] != label:
                    raise ValueError("conflicting labels for the same episode span")
                continue
            by_span[span] = label
            output.append((label, windows, features))
            if provenance is not None:
                provenance.append({"source": capture["source_sha256"], "channel": channel, "projected": projected})
            if projected:
                projection_counts[label] += 1
    return output


def choose_threshold(scores, truth, degraded):
    """Keep coverage subject to no false anomaly labels on calibration controls."""
    confidence = scores.max(axis=1)
    top = scores.argmax(axis=1)
    known = (truth > 0) & (truth < len(LABELS))
    accepted = ~degraded & (top > 0)
    false = accepted & (top != truth)
    minimum = np.nextafter(max(confidence[false], default=0.), np.inf)
    useful = accepted & known & (top == truth) & (confidence >= minimum)
    return float(min(confidence[useful], default=minimum))


def fit(manifest_path, output, seed=42):
    manifest = json.loads(manifest_path.read_text())
    if manifest["feature_set_version"] != VERSION:
        raise ValueError("contextual comparison uses traffic-v1; use --method random_forest for traffic-v2")
    parts = load_partitions(manifest, manifest_path.parent, contextual=True)
    limits = application_limits(
        [w for w, label in parts["calibration"] if label == NORMAL], manifest.get("application_limits"),
    )
    training = [row for row in episodes(manifest, manifest_path.parent, "train") if row[0] != UNKNOWN]
    calibration = episodes(manifest, manifest_path.parent, "calibration")
    if {row[0] for row in training} != set(LABELS) or {row[0] for row in calibration} != set((*LABELS, UNKNOWN)):
        raise ValueError("train needs all known classes and calibration additionally needs unknowns")
    raw_train = np.asarray([row[2] for row in training])
    raw_cal = np.asarray([row[2] for row in calibration])
    train_y = np.asarray([LABELS.index(row[0]) for row in training])
    cal_y = np.asarray([LABELS.index(row[0]) if row[0] != UNKNOWN else len(LABELS) for row in calibration])
    pre = fit_transform_parameters(raw_train)
    train_x, cal_x = transform(raw_train, pre), transform(raw_cal, pre)
    degraded = np.asarray([application_degraded(row[1], limits) for row in calibration])
    output.mkdir(parents=True)
    result = dict(dataset_manifest_sha256=sha(manifest_path), models={})
    candidates = models(seed)
    for method in METHODS:
        model = candidates[method]
        model.fit(train_x, train_y)
        cal_logits = uncalibrated(model, method, cal_x)
        known = cal_y < len(LABELS)
        temperature = fit_temperature(cal_logits[known], cal_y[known])
        bundle = dict(schema_version=2, method=method, labels=LABELS, features=FEATURES + list(APP_FEATURES),
                      statistics=["median", "iqr"], binding=manifest["binding"], preprocessing=pre, model=encode_model(method, model),
                      temperature=temperature, application_limits=limits)
        scores = probabilities(raw_cal, bundle)
        bundle["confidence_threshold"] = choose_threshold(scores, cal_y, degraded)
        path = output / f"{method}.json"
        write_json(path, bundle)
        result["models"][method] = dict(sha256=sha(path), calibration=Counter(
            f"{row[0]}:{classify(row[1], bundle)['label']}" for row in calibration))
    write_json(output / "manifest.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("random_forest", "comparison"), default="comparison")
    parser.add_argument("--leave-out-type", choices=LABELS[1:],
                        help="fit an experimental v5 artifact excluding one type from fitting and calibration")
    args = parser.parse_args()
    if args.method == "random_forest":
        from netwatch_ml.training.random_forest_episode import fit as fit_forest
        fit_forest(args.manifest, args.output, leave_out_type=args.leave_out_type)
    else:
        if args.leave_out_type:
            parser.error("--leave-out-type requires Random Forest")
        fit(args.manifest, args.output)


if __name__ == "__main__":
    main()

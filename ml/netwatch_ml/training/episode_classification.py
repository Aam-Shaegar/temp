"""Compare episode classifiers with capture-level separation and portable inference."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.neural_network import MLPClassifier

from netwatch_ml.models.detector import FEATURES, fit_transform_parameters, transform
from netwatch_ml.models.episode_classifier import (
    LABELS, UNKNOWN, classify, episode_features, journal_windows, load, merge_journal_pages, probabilities, softmax,
)
from netwatch_ml.training.baselines import sha


PARTS = ("train", "calibration", "regression", "fresh")
METHODS = ("logistic", "histogram", "mlp")


def write_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def features_for_episode(windows):
    return episode_features([
        {
            "feature_set_version": w["feature_set_version"],
            "entity_type": w["entity_type"],
            "capture_epoch": w["group"],
            "window_start_ns": w["start_ns"],
            "window_end_ns": w["end_ns"],
            "quality": w["quality"],
            "features": w["features"],
        }
        for w in windows
    ])


def load_episodes(path, expected, role=None):
    windows = defaultdict(list)
    captures = {}
    names = {}
    for line in path.read_text().splitlines():
        window = json.loads(line)
        group = window["group"].rsplit("/", 1)[-1]
        if group not in expected:
            continue
        if role is not None and window["role"] != role:
            raise ValueError(f"unexpected role for {group}")
        digest = window["source_sha256"]
        if names.setdefault(group, window["group"]) != window["group"]:
            raise ValueError(f"colliding capture name: {group}")
        if captures.setdefault(group, digest) != digest:
            raise ValueError(f"multiple sources for {group}")
        windows[group, window["phase"]].append(window)
    if captures.keys() != set(expected) or len(set(captures.values())) != len(captures):
        raise ValueError("missing or duplicate captures")
    episodes = []
    for (group, phase), records in sorted(windows.items()):
        records.sort(key=lambda w: w["start_ns"])
        labels = {w["label"] for w in records}
        if len(labels) != 1 or not labels <= set(LABELS):
            raise ValueError(f"invalid or mixed episode truth: {group}/{phase}")
        features = features_for_episode(records)
        if features is None:
            raise ValueError(f"incomplete episode: {group}/{phase}")
        episodes.append((group, phase, labels.pop(), records, features))
    for group in expected:
        if {item[2] for item in episodes if item[0] == group} != set(LABELS):
            raise ValueError(f"missing class in capture {group}")
    return episodes, captures


def arrays(episodes):
    return (
        np.asarray([e[4] for e in episodes]),
        np.asarray([LABELS.index(e[2]) for e in episodes]),
    )


def models(seed):
    return {
        "logistic": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        "histogram": HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=7, min_samples_leaf=3, class_weight="balanced",
            early_stopping=False, random_state=seed,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(16,), solver="lbfgs", alpha=0.1, max_iter=1000, random_state=seed,
        ),
    }


def uncalibrated(model, method, x):
    if method == "mlp":
        # The public multiclass probabilities are sufficient for temperature scaling.
        return np.log(np.maximum(model.predict_proba(x), np.finfo(float).tiny))
    return model.decision_function(x)


def encode_model(method, model):
    if method == "logistic":
        return {"coefficients": model.coef_.tolist(), "intercepts": model.intercept_.tolist()}
    if method == "mlp":
        return {"layers": [[w.tolist(), b.tolist()] for w, b in zip(model.coefs_, model.intercepts_, strict=True)]}
    return {
        "intercepts": model._baseline_prediction.ravel().tolist(),
        "stages": [
            [
                [[int(n["feature_idx"]), float(n["num_threshold"]),
                  -1 if n["is_leaf"] else int(n["left"]), -1 if n["is_leaf"] else int(n["right"]),
                  float(n["value"])] for n in tree.nodes]
                for tree in stage
            ]
            for stage in model._predictors
        ],
    }


def fit_temperature(logits, y):
    result = minimize_scalar(
        lambda log_t: log_loss(y, softmax(logits, np.exp(log_t)), labels=list(range(len(LABELS)))),
        bounds=(np.log(0.25), np.log(4.0)), method="bounded",
    )
    if not result.success:
        raise RuntimeError("calibration did not converge")
    return float(np.exp(result.x))


def abstention_threshold(values, y):
    """Maximize accepted coverage subject to no observed calibration errors."""
    chosen = values.argmax(axis=1)
    confidence = values.max(axis=1)
    safe = []
    for threshold in sorted(set(confidence)):
        keep = confidence >= threshold
        if not np.any(chosen[keep] != y[keep]):
            if len(set(chosen[keep])) == len(LABELS):
                return float(threshold)
            safe.append(float(threshold))
    return safe[0] if safe else float(np.nextafter(1.0, 2.0))


def inverse_frequency(y):
    return len(y) / (len(LABELS) * np.bincount(y)[y])


def train(input_path, plan_path, output):
    plan = json.loads(plan_path.read_text())
    groups = {name: set(plan[f"{name}_groups"]) for name in PARTS}
    if sum(map(len, groups.values())) != len(set().union(*groups.values())):
        raise ValueError("partitions share a capture")
    train_rows, train_captures = load_episodes(input_path, groups["train"], "development_train_candidate")
    calibration_rows, calibration_captures = load_episodes(
        input_path, groups["calibration"], "development_train_candidate",
    )
    if set(train_captures.values()) & set(calibration_captures.values()):
        raise ValueError("training and calibration share source bytes")
    raw_train, train_y = arrays(train_rows)
    raw_cal, cal_y = arrays(calibration_rows)
    preprocessing = fit_transform_parameters(raw_train)
    train_x, cal_x = transform(raw_train, preprocessing), transform(raw_cal, preprocessing)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "dataset_sha256": sha(input_path), "protocol_sha256": sha(plan_path),
        "sklearn_version": sklearn.__version__, "feature_set_version": "traffic-v1",
        "train": train_captures, "calibration": calibration_captures,
        "parts": {"train": len(train_rows), "calibration": len(calibration_rows)},
        "models": {},
    }
    for method, model in models(plan["seed"]).items():
        weights = inverse_frequency(train_y) if method == "mlp" else None
        model.fit(train_x, train_y, sample_weight=weights)
        cal_logits = uncalibrated(model, method, cal_x)
        temperature = fit_temperature(cal_logits, cal_y)
        cal_p = softmax(cal_logits, temperature)
        threshold = abstention_threshold(cal_p, cal_y)
        bundle = {
            "schema_version": 1, "method": method, "labels": LABELS,
            "features": FEATURES, "statistics": ["median", "iqr"],
            "preprocessing": preprocessing, "temperature": temperature,
            "confidence_threshold": threshold, "model": encode_model(method, model),
        }
        for raw, x in ((raw_train, train_x), (raw_cal, cal_x)):
            reference = softmax(uncalibrated(model, method, x), temperature)
            np.testing.assert_allclose(probabilities(raw, bundle), reference, atol=1e-10, rtol=1e-9)
        location = output / f"{method}.json"
        write_json(location, bundle)
        manifest["models"][method] = {
            "sha256": sha(location),
            "calibration_accepted": {
                label: int(((cal_y == i) & (cal_p.max(axis=1) >= threshold)).sum())
                for i, label in enumerate(LABELS)
            },
        }
    write_json(output / "manifest.json", manifest)


def metrics(episodes, bundle):
    raw, truth = arrays(episodes)
    scores = probabilities(raw, bundle)
    top = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    decision = np.where(confidence >= bundle["confidence_threshold"], top, len(LABELS))
    names = (*LABELS, UNKNOWN)
    cases = []
    matrix = {label: Counter() for label in LABELS}
    for row, expected, estimate, selected in zip(episodes, truth, top, decision, strict=True):
        group, phase, _, windows, _ = row
        predicted = names[selected]
        matrix[LABELS[expected]][predicted] += 1
        cases.append({
            "group": group, "phase": phase, "actual": LABELS[expected],
            "predicted": predicted, "top_label": LABELS[estimate], "windows": len(windows),
        })
    per_class = {}
    for name in LABELS:
        count = sum(matrix[name].values())
        hits = matrix[name][name]
        predicted = sum(m[name] for m in matrix.values())
        per_class[name] = {
            "support": count, "correct": hits, "unknown": matrix[name][UNKNOWN],
            "recall": hits / count, "precision": hits / predicted if predicted else None,
        }
    accepted = decision != len(LABELS)
    return {
        "total": len(truth), "top1_correct": int((top == truth).sum()),
        "correct": int((decision == truth).sum()),
        "rejected": int((~accepted).sum()),
        "incorrect_accepted": int((accepted & (decision != truth)).sum()),
        "false_positive_normal": sum(matrix[LABELS[0]][label] for label in LABELS[1:]),
        "macro_f1": float(f1_score(truth, decision, labels=list(range(len(LABELS))), average="macro")),
        "top1_macro_f1": float(f1_score(truth, top, labels=list(range(len(LABELS))), average="macro")),
        "confusion": matrix, "per_class": per_class, "episodes": cases,
    }


def evaluate(input_path, plan_path, models_path, report, fresh=None):
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((models_path / "manifest.json").read_text())
    if sha(input_path) != manifest["dataset_sha256"] or sha(plan_path) != manifest["protocol_sha256"]:
        raise ValueError("training data or protocol changed after model freeze")
    baseline, baseline_captures = load_episodes(input_path, set(plan["regression_groups"]), "regression_only")
    datasets = {"regression": baseline}
    provenance = {"regression": baseline_captures}
    if fresh is not None:
        datasets["fresh"], provenance["fresh"] = load_episodes(fresh, set(plan["fresh_groups"]))
    splits = [manifest["train"], manifest["calibration"], *provenance.values()]
    if sum(map(len, splits)) != len({digest for part in splits for digest in part.values()}):
        raise ValueError("source overlap among training, calibration and evaluation")
    result = {
        "provenance": provenance, "models": {}, "model_manifest_sha256": sha(models_path / "manifest.json"),
        "fresh_windows_sha256": sha(fresh) if fresh else None,
    }
    for method in METHODS:
        path = models_path / f"{method}.json"
        bundle = load(path, manifest["models"][method]["sha256"])
        if bundle["method"] != method:
            raise ValueError(f"frozen model identity mismatch: {method}")
        result["models"][method] = {kind: metrics(episodes, bundle) for kind, episodes in datasets.items()}
    write_json(report, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    score = commands.add_parser("evaluate")
    for command in (fit, score):
        command.add_argument("input", type=Path)
        command.add_argument("protocol", type=Path)
        command.add_argument("models", type=Path)
    score.add_argument("report", type=Path)
    score.add_argument("--fresh", type=Path)
    infer = commands.add_parser("classify")
    infer.add_argument("windows", type=Path, help="one episode of native Go windows in JSONL")
    infer.add_argument("bundle", type=Path)
    infer.add_argument("sha256")
    infer.add_argument("--journal", action="store_true",
                       help="read an operator incident JSON response or an ordered JSON array of its pages")
    args = parser.parse_args()
    if args.command == "train":
        train(args.input, args.protocol, args.models)
    elif args.command == "evaluate":
        evaluate(args.input, args.protocol, args.models, args.report, args.fresh)
    else:
        from netwatch_ml.contracts import validate

        bundle = load(args.bundle, args.sha256)
        body = args.windows.read_text()
        channel = "combined"
        if args.journal:
            journal = json.loads(body)
            if isinstance(journal, list):
                journal = merge_journal_pages(journal)
            rows = journal_windows(journal)
            channel = journal["incident"].get("evidence_channel", "combined")
        else:
            rows = map(json.loads, body.splitlines())
        print(json.dumps(classify([validate("window", row) for row in rows], bundle, evidence_channel=channel)))


if __name__ == "__main__":
    main()

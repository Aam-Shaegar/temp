"""Recalculate event metrics from archived logs without running the models."""

import argparse
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .event_evaluator import evaluate_groups
else:
    from event_evaluator import evaluate_groups


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reported_metrics(previous):
    detector = previous["detector"]
    classifier = previous["classifier"]["random_forest"]
    return dict(
        known_found=detector["known_found"], unknown_found=detector["unknown_found"],
        false_alarms=detector.get("false_alerts_including_bad_boundaries", detector["false_incidents"]),
        pure_false_alarms=detector["false_incidents"], split=detector.get("split"), merged=detector["merged"],
        known_correct=classifier["known_correct"], macro_recall=classifier["macro_recall"],
    )


def comparable(metrics):
    return dict((k, metrics[k]) for k in ("known_found", "unknown_found", "false_alarms", "pure_false_alarms", "split", "merged")) | {
        "known_correct": metrics["classifier"]["known_correct"], "macro_recall": metrics["classifier"]["macro_recall"],
    }


def compare_values(current, previous):
    previous = reported_metrics(previous)
    current = comparable(current)
    return dict(archived=previous, recalculated=current, changes={
        name: value - previous[name] for name, value in current.items()
        if value is not None and previous[name] is not None
        and not math.isclose(value, previous[name], rel_tol=0, abs_tol=1e-15)
    })


def recalculate(folder, protocol_path):
    inputs = {name: folder / (name + ".json") for name in ("episodes", "journal-expected", "trial")}
    episodes, journal, trial = (read(path) for path in inputs.values())
    protocol = read(protocol_path)
    variants = trial["results"]
    keys = {
        "known_recall": "min_recall",
        "false_incidents_per_normal_hour": "max_false_per_hour",
        "min_observation_coverage": "min_evaluated_fraction",
    }
    limits = {keys.get(key, key): value for key, value in protocol["detector_limits"].items()}
    limits.update(protocol["classifier_limits"])
    # The saved test cases carry the service name in their path, independent of prediction order.
    services = {}
    for path in sorted((folder / "prepared").glob("*/*.test.json")):
        case_id = path.name.removesuffix(".test.json")
        if case_id in services:
            raise ValueError(f"Duplicate archived test case: {case_id}")
        services[case_id] = path.parent.name
    if set(variants) != set(episodes) or not services:
        raise ValueError("Incomplete archived results")

    results = {}
    for variant, cases in sorted(episodes.items()):
        expected = journal[f"{variant}/{protocol['policy']}"]
        previous = variants[variant]
        if cases.keys() != services.keys() or cases.keys() != expected.keys():
            raise ValueError(f"Missing archived capture for {variant}")
        groups = []
        for case_id, case in sorted(cases.items()):
            incidents = case["incidents"]
            evidence = expected[case_id]
            if len(incidents) != len(evidence) or any(
                any(incident[key] != value for key, value in raw.items())
                for incident, raw in zip(incidents, evidence, strict=True)
            ):
                raise ValueError(f"Stored predictions do not match the archived journal for {variant}/{case_id}")
            groups.append(dict(
                id=case_id, service=services[case_id], incidents=[
                    {key: value for key, value in incident.items() if key not in {"matched", "merged", "starts_normal", "pure_false"}}
                    for incident in incidents
                ],
                truth=[{key: value for key, value in p.items() if key != "incident"} for p in case["truth"]],
            ))
        # The archive does not retain individual window availability for each model.
        # Original aggregate window metrics are copied as provenance, never recomputed.
        labels = previous["classifier"]["random_forest"]["per_class"]
        result = evaluate_groups(groups, limits, start_tolerance_s=5, known_labels=labels, classifier="random_forest")
        result["archived_window_metrics"] = previous["windows"]
        result["comparison"] = dict(
            overall=compare_values(result["overall"], previous),
            by_service={s: compare_values(metrics, previous["by_service"][s]) for s, metrics in result["by_service"].items()},
        )
        results[variant] = result
    return dict(
        schema_version=1, mode="archived_predictions", inference_performed=False,
        window_availability="Missing per-window states. Labelled normal hours are exploratory exposure only.",
        source_sha256={name: digest(path) for name, path in inputs.items()} | {"protocol": digest(protocol_path)},
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "protocol", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = recalculate(args.results, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

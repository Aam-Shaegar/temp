"""Score frozen incidents against event-time truth with one shared reducer."""

from collections import Counter, defaultdict, deque
import math
import statistics

NS = 10**9
HOUR_NS = 3600 * NS
NON_KNOWN = {"normal", "unknown", "mixed", "unobservable"}
EVIDENCE_CHANNELS = {"combined", "network", "http"}
REGIMES = ("planned", "unplanned", "unspecified")
TYPED_REJECTION = {"low_confidence", "application_degradation", "unknown_type"}
DEFAULT_LIMITS = dict(
    min_recall=0.8,
    max_false_per_hour=1,
    max_start_error_s=15,
    max_end_error_s=15,
    min_normal_hours=8,
    known_macro_recall=0.8,
    unknown_abstention=0.8,
    false_labels_per_normal_hour=1,
)


def overlap(start, end, other_start, other_end):
    return max(0, min(end, other_end) - max(start, other_start))


def compatible_channel(truth, incident):
    reference = truth.get("evidence_channel")
    return reference is None or incident.get("evidence_channel", "combined") in (reference, "combined")


def maximum_matches(positive_count, incident_count, edges):
    """Find maximum cardinality, then maximum total overlap, using integer costs.

    Each shortest residual augmenting path adds a match and can reassign
    earlier matches. Stable vertex and edge traversal breaks equal-cost ties.
    """
    source = 0
    left = 1
    right = left + positive_count
    sink = right + incident_count
    graph = [[] for _ in range(sink + 1)]

    def connect(u, v, cost):
        edge = [v, cost, 1, len(graph[v])]
        graph[v].append([u, -cost, 0, len(graph[u])])
        graph[u].append(edge)
        return edge

    for i in range(positive_count):
        connect(source, left + i, 0)
    links = {(i, j): connect(left + i, right + j, -amount) for i, j, amount in sorted(edges)}
    for j in range(incident_count):
        connect(right + j, sink, 0)

    while True:
        dist, previous = [None] * len(graph), [None] * len(graph)
        pending, queued = deque([source]), {source}
        dist[source] = 0
        while pending:
            u = pending.popleft()
            queued.remove(u)
            for index, (v, cost, free, _) in enumerate(graph[u]):
                if not free or v == source:
                    continue
                distance = dist[u] + cost
                if dist[v] is None or distance < dist[v]:
                    dist[v] = distance
                    previous[v] = u, index
                    if v != sink and v not in queued:
                        pending.append(v)
                        queued.add(v)
        if previous[sink] is None:
            break
        v = sink
        while v != source:
            u, index = previous[v]
            edge = graph[u][index]
            edge[2] = 0
            graph[v][edge[3]][2] = 1
            v = u
    return dict((i, j) for (i, j), edge in links.items() if edge[2] == 0)


def normal_regime(row):
    if type(row.get("planned")) is bool:
        return "planned" if row["planned"] else "unplanned"
    return row.get("regime", "unspecified")


def reference_intervals(group):
    truth = sorted(group["truth"], key=lambda row: (row["start"], row["end"], row["label"]))
    if any(type(row[key]) is not int for row in truth for key in ("start", "end")) or any(
        row["end"] <= row["start"] for row in truth
    ):
        raise ValueError("Truth requires nonempty event-time intervals in integer nanoseconds")
    if any(not isinstance(p["label"], str) or not p["label"] for p in truth):
        raise ValueError("Every truth interval requires a label")
    if any(p.get("evidence_channel", "combined") not in EVIDENCE_CHANNELS for p in truth):
        raise ValueError("Unrecognized truth evidence channel")
    normal_rows = [p for p in truth if p["label"] == "normal"]
    if any(normal_regime(p) not in REGIMES for p in normal_rows):
        raise ValueError("Unrecognized normal regime")
    normal = [(p["start"], p["end"]) for p in normal_rows]
    positive = [p for p in truth if p["label"] not in {"normal", "unobservable"}]
    if any(end > next_start for (_, end), (next_start, _) in zip(normal, normal[1:])) or any(
        overlap(start, end, p["start"], p["end"])
        for start, end in normal for p in truth if p["label"] != "normal"
    ):
        raise ValueError("Normal intervals must not overlap other truth")
    return normal_rows, positive


def record_group(group, start_tolerance_s):
    normal, positive = reference_intervals(group)
    incident_rows = sorted(group["incidents"], key=lambda row: (row["first"], row["detected"], str(row.get("id", ""))))
    if any(type(row[key]) is not int for row in incident_rows for key in ("first", "last", "detected")) or any(
        row.get("end") is not None and type(row["end"]) is not int for row in incident_rows
    ):
        raise ValueError("Incidents require integer event-time boundaries")
    if any(row.get("evidence_channel", "combined") not in EVIDENCE_CHANNELS for row in incident_rows):
        raise ValueError("Unrecognized incident evidence channel")
    tolerance_ns = None if start_tolerance_s is None else round(start_tolerance_s * NS)
    overlaps_by_truth, overlaps_by_incident = Counter(), Counter()
    any_positive_overlap = set()
    eligible = []
    for i, p in enumerate(positive):
        for j, event in enumerate(incident_rows):
            end = event["last"] if event.get("end") is None else event["end"]
            intersection = overlap(p["start"], p["end"], event["first"], end)
            if not intersection:
                continue
            any_positive_overlap.add(j)
            if not compatible_channel(p, event):
                continue
            overlaps_by_truth[i] += 1
            overlaps_by_incident[j] += 1
            if tolerance_ns is None:
                allowed = p["start"] <= event["detected"] < p["end"]
            else:
                allowed = p["start"] - tolerance_ns <= event["first"] < p["end"]
            if allowed:
                eligible.append((i, j, intersection))
    matches = maximum_matches(len(positive), len(incident_rows), eligible)
    used = set(matches.values())
    incidents = []
    for j, row in enumerate(incident_rows):
        matched = j in used
        record = dict(row, matched=matched)
        normal_at_confirmation = next(
            (p for p in normal if p["start"] <= row["detected"] < p["end"]), None,
        )
        record["false_alarm"] = not matched and normal_at_confirmation is not None
        record["pure_false_alarm"] = record["false_alarm"] and j not in any_positive_overlap
        record["false_alarm_regime"] = normal_regime(normal_at_confirmation) if record["false_alarm"] else None
        incidents.append(record)
    labelled_by_regime = Counter()
    for row in normal:
        labelled_by_regime[normal_regime(row)] += row["end"] - row["start"]
    labelled = sum(labelled_by_regime.values())
    observation = group.get("observation")
    if observation is not None:
        windows = observation["windows"]
        evaluated = observation["evaluated"]
        verified = observation["normal_evaluated_ns"]
        if not all(type(value) is int for value in (windows, evaluated, verified)) or not (
            0 <= evaluated <= windows and 0 <= verified <= labelled
        ):
            raise ValueError("Invalid observation coverage")
        by_regime = observation.get("normal_evaluated_ns_by_regime")
        if by_regime is not None and (
            set(by_regime) - set(REGIMES)
            or any(type(duration) is not int or not 0 <= duration <= labelled_by_regime[regime]
                   for regime, duration in by_regime.items())
            or sum(by_regime.values()) != verified
        ):
            raise ValueError("Invalid observation coverage by regime")
    return dict(
        id=group["id"], service=group["service"], normal_labelled_ns=labelled,
        normal_labelled_ns_by_regime=dict(labelled_by_regime),
        observation=observation, incidents=incidents,
        positive=[dict(p, match=matches.get(i), overlaps=overlaps_by_truth[i]) for i, p in enumerate(positive)],
        unobservable_intervals=sum(p["label"] == "unobservable" for p in group["truth"]),
        split=sum(n > 1 for n in overlaps_by_truth.values()),
        merged=sum(n > 1 for n in overlaps_by_incident.values()),
    )


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, level):
    if not values:
        return None
    ordered = sorted(values)
    index = level * (len(ordered) - 1)
    left, right = math.floor(index), math.ceil(index)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def timing_seconds(values):
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                p50=percentile(values, 0.5), p95=percentile(values, 0.95), max=max(values, default=None))


def poisson_upper(count, hours):
    """One-sided 95% Poisson upper rate, conditional on a homogeneous process."""
    if not hours:
        return None
    target = math.log(0.05)
    if count == 0:
        return -target / hours
    log_factorials = [math.lgamma(i + 1) for i in range(count + 1)]
    low, high = 0.0, count + 10 * math.sqrt(count + 1) + 10
    for _ in range(48):
        mean = (low + high) / 2
        log_terms = [i * math.log(mean) - mean - term for i, term in enumerate(log_factorials)]
        maximum = max(log_terms)
        log_cdf = maximum + math.log(sum(math.exp(term - maximum) for term in log_terms))
        if log_cdf > target:
            low = mean
        else:
            high = mean
    return high / hours


def type_score(records, labels, classifier, hours, limits):
    actual = Counter()
    predicted = Counter()
    correct = Counter()
    outcome_by_label = defaultdict(Counter)
    unknown_total = Counter()
    unknown_rejected = Counter()
    matched_known = 0
    accepted_known = 0
    false_labels = 0
    for group in records:
        incidents = group["incidents"]
        for incident in incidents:
            label = incident.get("outputs", {}).get(classifier, {}).get("label")
            if label in labels:
                predicted[label] += 1
                false_labels += incident["false_alarm"]
        for truth in group["positive"]:
            label, match = truth["label"], truth["match"]
            channel = truth.get("evidence_channel", "combined")
            if label in labels:
                actual[label] += 1
                matched_known += match is not None
            elif label == "unknown":
                unknown_total[channel] += 1
            result = incidents[match].get("outputs", {}).get(classifier, {}) if match is not None else {}
            prediction = result.get("label")
            if match is None:
                outcome = "not_detected"
            elif prediction is None:
                outcome = "unavailable"
            elif prediction == "unknown":
                reason = result.get("reason", "unspecified")
                outcome = f"unknown:{reason}"
                typed_rejection = reason in TYPED_REJECTION
                if reason == "application_degradation" and channel == "network":
                    typed_rejection = False
                if label == "unknown" and typed_rejection:
                    unknown_rejected[channel] += 1
            else:
                outcome = str(prediction)
            outcome_by_label[label][outcome] += 1
            if label in labels:
                accepted_known += prediction in labels
                correct[label] += prediction == label
    per_class = {
        label: dict(total=actual[label], hits=correct[label], predictions=predicted[label],
                    recall=ratio(correct[label], actual[label]), precision=ratio(correct[label], predicted[label]),
                    f1=ratio(2 * correct[label], predicted[label] + actual[label]), outcomes=dict(outcome_by_label[label]))
        for label in labels
    }
    present = [m for m in per_class.values() if m["total"]]
    macro_recall = statistics.mean(m["recall"] for m in present) if present and len(present) == len(labels) else None
    macro_f1 = statistics.mean(m["f1"] for m in present) if present and len(present) == len(labels) else None
    unknown_abstention = ratio(unknown_rejected.total(), unknown_total.total())
    unknown_by_channel = {
        channel: dict(total=unknown_total[channel], rejected=unknown_rejected[channel],
                      recall=ratio(unknown_rejected[channel], unknown_total[channel]))
        for channel in sorted(EVIDENCE_CHANNELS)
    }
    false_typed_per_hour = ratio(false_labels, hours)
    gate = dict(
        known_recall=macro_recall is not None and macro_recall >= limits["known_macro_recall"],
        false_labels=false_typed_per_hour is not None and false_typed_per_hour <= limits["false_labels_per_normal_hour"],
        unknown=unknown_abstention is not None and unknown_abstention >= limits["unknown_abstention"],
    )
    if limits.get("require_network_unknown", False):
        network_abstention = unknown_by_channel["network"]["recall"]
        gate["network_unknown"] = network_abstention is not None and network_abstention >= limits["unknown_abstention"]
    return dict(per_class=per_class, macro_recall=macro_recall, macro_f1=macro_f1,
                known_total=sum(actual.values()), known_correct=sum(correct.values()),
                known_matched=matched_known, known_accepted=accepted_known,
                conditional_typing_recall=ratio(sum(correct.values()), matched_known),
                coverage=ratio(accepted_known, sum(actual.values())),
                accepted_risk=ratio(accepted_known - sum(correct.values()), accepted_known),
                false_labels=false_labels, false_typed_per_hour=false_typed_per_hour,
                unknown_total=unknown_total.total(), explicit_unknown=unknown_rejected.total(),
                unknown_abstention=unknown_abstention, unknown_by_channel=unknown_by_channel,
                unknown_outcomes=dict(outcome_by_label["unknown"]),
                unknown_overclaimed=sum(outcome_by_label["unknown"][label] for label in labels),
                mixed_truth_outcomes=dict(outcome_by_label["mixed"]),
                gate=gate, passed=all(gate.values()))


def reduce_groups(records, labels, classifier, limits):
    known = Counter()
    found = Counter()
    start_errors, end_errors, confirmation_delays, durations = [], [], [], []
    unavailable = Counter()
    labelled_by_regime, verified_by_regime = Counter(), Counter()
    false_by_regime, pure_false_by_regime, false_by_channel = Counter(), Counter(), Counter()
    known_by_regime, found_by_regime = Counter(), Counter()
    regime_observed = 0
    total = Counter()
    for group in records:
        total["normal_labelled_ns"] += group["normal_labelled_ns"]
        labelled_by_regime.update(group["normal_labelled_ns_by_regime"])
        observation = group["observation"]
        if observation is not None:
            total["observed_captures"] += 1
            for field in ("windows", "evaluated", "normal_evaluated_ns"):
                total[field] += observation[field]
            unavailable.update(observation.get("unavailable", {}))
            split_exposure = observation.get("normal_evaluated_ns_by_regime")
            if split_exposure is None and observation["normal_evaluated_ns"] == group["normal_labelled_ns"]:
                split_exposure = group["normal_labelled_ns_by_regime"]
            if split_exposure is not None:
                verified_by_regime.update(split_exposure)
                regime_observed += 1
        total["unobservable_intervals"] += group["unobservable_intervals"]
        total["split"] += group["split"]
        total["merged"] += group["merged"]
        incidents = group["incidents"]
        total["false_alarms"] += sum(row["false_alarm"] for row in incidents)
        total["pure_false_alarms"] += sum(row["pure_false_alarm"] for row in incidents)
        for incident in incidents:
            if incident["false_alarm"]:
                regime = incident["false_alarm_regime"]
                false_by_regime[regime] += 1
                pure_false_by_regime[regime] += incident["pure_false_alarm"]
                false_by_channel[incident.get("evidence_channel", "combined")] += 1
        total["unmatched_incidents"] += sum(not row["matched"] for row in incidents)
        total["interrupted"] += sum(row.get("interrupted") is not None for row in incidents)
        for p in group["positive"]:
            known[p["label"]] += 1
            match = p["match"]
            if p["label"] in labels:
                regime = normal_regime(p)
                known_by_regime[regime] += 1
                found_by_regime[regime] += match is not None
            if match is None:
                total["overlaps_without_match"] += p["overlaps"] > 0
                continue
            found[p["label"]] += 1
            row = incidents[match]
            start_errors.append(abs(row["first"] - p["start"]) / NS)
            confirmation_delays.append((row["detected"] - p["start"]) / NS)
            if row.get("end") is None:
                total["unresolved_ends"] += 1
            else:
                end_errors.append(abs(row["end"] - p["end"]) / NS)
            if row.get("closed") is not None:
                durations.append((row["closed"] - p["end"]) / NS)
    verified = bool(records) and total["observed_captures"] == len(records)
    hours = (total["normal_evaluated_ns"] if verified else total["normal_labelled_ns"]) / HOUR_NS
    hours_kind = "evaluated" if verified else "labelled_only"
    false_rate = ratio(total["false_alarms"], hours)
    known_total = sum(known[label] for label in labels)
    known_found = sum(found[label] for label in labels)
    known_recall = ratio(known_found, known_total)
    coverage = ratio(total["evaluated"], total["windows"]) if verified else None
    gate = dict(
        recall=known_recall is not None and known_recall >= limits["min_recall"],
        false_rate=false_rate is not None and false_rate <= limits["max_false_per_hour"],
        normal_exposure=verified and hours >= limits["min_normal_hours"],
        start_error=bool(start_errors) and max(start_errors) <= limits["max_start_error_s"],
        end_error=bool(end_errors) and total["unresolved_ends"] == 0 and max(end_errors) <= limits["max_end_error_s"],
        episode_integrity=not (total["split"] or total["merged"]),
    )
    if "min_evaluated_fraction" in limits:
        gate["scoring_coverage"] = coverage is not None and coverage >= limits["min_evaluated_fraction"]
    minimum_support = limits.get("min_episodes_per_known_class")
    if minimum_support is not None:
        gate["known_support"] = bool(labels) and all(known[label] >= minimum_support for label in labels)
    for regime, minimum in limits.get("min_known_by_regime", {}).items():
        gate[f"{regime}_network_support"] = known_by_regime[regime] >= minimum
        regime_recall = ratio(found_by_regime[regime], known_by_regime[regime])
        gate[f"{regime}_network_recall"] = regime_recall is not None and regime_recall >= limits["min_recall"]
    regime_verified = verified and regime_observed == len(records)
    regime_metrics = {}
    for regime in REGIMES:
        if labelled_by_regime[regime] or false_by_regime[regime]:
            measured = verified_by_regime[regime] / HOUR_NS if regime_verified else None
            regime_metrics[regime] = dict(
                normal_labelled_hours=labelled_by_regime[regime] / HOUR_NS,
                normal_evaluated_hours=measured, false_alarms=false_by_regime[regime],
                pure_false_alarms=pure_false_by_regime[regime],
                false_per_hour=ratio(false_by_regime[regime], measured),
            )
    result = dict(
        captures=len(records), known_total=known_total, known_found=known_found, known_recall=known_recall,
        unknown_total=known["unknown"], unknown_found=found["unknown"],
        per_class={label: dict(total=known[label], found=found[label], recall=ratio(found[label], known[label]))
                   for label in (*labels, "unknown")},
        known_by_regime={regime: dict(total=known_by_regime[regime], found=found_by_regime[regime],
                                      recall=ratio(found_by_regime[regime], known_by_regime[regime]))
                         for regime in REGIMES},
        normal_labelled_hours=total["normal_labelled_ns"] / HOUR_NS,
        normal_evaluated_hours=hours if verified else None,
        normal_hours=hours, normal_hours_kind=hours_kind, false_alarms=total["false_alarms"],
        false_per_hour=false_rate, pure_false_alarms=total["pure_false_alarms"],
        by_normal_regime=regime_metrics, false_alarms_by_evidence_channel=dict(false_by_channel),
        poisson_upper_95_per_hour=poisson_upper(total["false_alarms"], hours),
        unmatched_incidents=total["unmatched_incidents"], overlaps_without_match=total["overlaps_without_match"],
        unobservable_truth_intervals=total["unobservable_intervals"],
        split=total["split"], merged=total["merged"], interrupted=total["interrupted"],
        unresolved_ends=total["unresolved_ends"], windows=total["windows"] if verified else None,
        evaluated=total["evaluated"] if verified else None, evaluated_fraction=coverage,
        unavailable=dict(unavailable) if verified else None,
        start_error_s=timing_seconds(start_errors), end_error_s=timing_seconds(end_errors),
        opening_delay_s=timing_seconds(confirmation_delays), closing_delay_s=timing_seconds(durations),
        checks=gate, passed=all(gate.values()),
    )
    if classifier is not None:
        score = type_score(records, labels, classifier, hours, limits)
        result["classifier"] = score
        result["passed"] = result["passed"] and score["passed"]
    return result


def evaluate_groups(groups, limits=None, *, start_tolerance_s=5, known_labels=None, classifier=None):
    """Return event metrics overall and by service under the same definition.

    Each group has id, service, truth, incidents and optionally observation
    (counts of scored/total windows and verified normal duration in ns).
    A None start tolerance retains legacy confirmation-in-truth matching.
    """
    effective_limits = DEFAULT_LIMITS | (limits or {})
    records = [record_group(group, start_tolerance_s) for group in groups]
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("Expected nonempty capture groups with unique identifiers")
    labels = (tuple(sorted({p["label"] for r in records for p in r["positive"]} - NON_KNOWN))
              if known_labels is None else tuple(sorted(set(known_labels))))
    services = defaultdict(list)
    for record in records:
        services[record["service"]].append(record)
    return dict(
        matching_version="max-cardinality-max-overlap-v1", start_tolerance_s=start_tolerance_s,
        limits=effective_limits, known_labels=labels,
        overall=reduce_groups(records, labels, classifier, effective_limits),
        by_service={service: reduce_groups(services[service], labels, classifier, effective_limits)
                    for service in sorted(services)},
        cases=records,
    )

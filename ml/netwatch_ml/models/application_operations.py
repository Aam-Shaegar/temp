"""Shared numeric features from independently measured HTTP operations."""

OPERATION_FEATURES = (
    "http_mean_request_body_bytes",
    "http_max_operation_mean_request_body_bytes",
    "http_mean_response_body_bytes",
    "http_max_operation_mean_response_body_bytes",
    "http_expected_requests",
    "http_missing_requests",
    "http_excess_requests",
)


def operation_features(application):
    """Ignore identity fields and preserve unknown operation coverage and volume."""
    if not application or not (operations := application.get("operations")):
        return [None] * len(OPERATION_FEATURES)
    requests = application["completed_requests"]
    if sum(row["completed_requests"] for row in operations) != requests:
        return [None] * len(OPERATION_FEATURES)
    observed = [row for row in operations if row["completed_requests"]]
    features = []
    for name in ("request_body_bytes", "response_body_bytes"):
        if observed and all(row[name] is not None for row in observed):
            features.extend((sum(row[name] for row in observed) / requests,
                             max(row[name] / row["completed_requests"] for row in observed)))
        else:
            features.extend((None, None))
    if all(row["expected_requests"] is not None for row in operations):
        features.extend((sum(row["expected_requests"] for row in operations),
                         sum(max(row["expected_requests"] - row["completed_requests"], 0) for row in operations),
                         sum(max(row["completed_requests"] - row["expected_requests"], 0) for row in operations)))
    else:
        features.extend((None, None, None))
    return features


def operation_measurement_available(application):
    """Use the same measurement boundary for fitting and online scoring."""
    return any(value is not None for value in operation_features(application))

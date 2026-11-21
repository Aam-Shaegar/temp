import json
import os
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.aggregate_http_context import WIDTH, aggregate, publish  # noqa: E402


def records(path, *items):
    path.write_text("".join(json.dumps(row) + "\n" for row in items))
    return path


def completion(stamp, route="catalogue", **fields):
    return dict(dict(completed_at_ns=str(stamp), route_id=route, method="GET", status=200,
                     latency_ms=3.0, request_body_bytes=0, response_body_bytes=500), **fields)


def expectation(stamp, as_of, route="catalogue", **fields):
    return dict(dict(window_start_ns=str(stamp), as_of_ns=str(as_of), route_id=route, method="GET",
                     expected_requests=4), **fields)


def test_completion_buckets_keep_unknown_sizes_and_observed_demand_gaps(tmp_path):
    events = records(tmp_path / "events", completion(0), completion(WIDTH - 1, status=503, response_body_bytes=None),
                     completion(WIDTH, "download"))
    expected = records(tmp_path / "predicted", expectation(0, 0), expectation(WIDTH, 0),
                       expectation(2 * WIDTH, 0, expected_requests=0))
    schedule = tmp_path / "schedule"
    schedule.write_text(json.dumps([dict(start_ns=WIDTH, end_ns=2 * WIDTH)]))
    first, second, third = aggregate(events, 0, 3 * WIDTH, "sensor", "TCP/127.0.0.1:8080",
                                     expectations=expected, planned=schedule, source="log-1")
    assert first["completed_requests"] == 2 and first["server_errors"] == 1
    assert first["planned_activity"] is False and second["planned_activity"] is True
    assert first["source_ref"] == "log-1.0"
    assert first["operations"][0] == dict(route_id="catalogue", method="GET", completed_requests=2,
                                          request_body_bytes=0, response_body_bytes=None, expected_requests=4)
    assert second["operations"][0]["completed_requests"] == 0
    assert second["operations"][0]["expected_requests"] == 4
    assert second["operations"][1]["response_body_bytes"] == 500
    assert third["completed_requests"] == 0 and third["mean_latency_ms"] is None
    assert third["operations"][0]["expected_requests"] == 0


def test_missing_calendar_and_publishing_the_same_snapshot(tmp_path):
    events = records(tmp_path / "events", completion(0))
    row, = aggregate(events, 0, WIDTH, "sensor", "TCP/127.0.0.1:8080")
    assert row["planned_activity"] is None and "source_ref" not in row
    publish(tmp_path, row)
    publish(tmp_path, row)
    with pytest.raises(ValueError, match="previously exported"):
        publish(tmp_path, dict(row, completed_requests=2))
    assert json.loads((tmp_path / "0.json").read_text()) == row


def test_export_permissions_are_private_by_default_and_shared_by_request(tmp_path):
    events = records(tmp_path / "events", completion(0))
    first, second = aggregate(events, 0, 2 * WIDTH, "sensor", "TCP/127.0.0.1:8080")
    publish(tmp_path, first)
    private = tmp_path / "0.json"
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    gid = os.getgid()
    with pytest.raises(ValueError, match="permissions differ"):
        publish(tmp_path, first, gid)
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    publish(tmp_path, second, gid)
    publish(tmp_path, second, gid)
    shared = (tmp_path / f"{WIDTH}.json").stat()
    assert shared.st_gid == gid and stat.S_IMODE(shared.st_mode) == 0o640
    assert not list(tmp_path.glob(".http-context-*"))


def test_future_demand_or_uncovered_transaction_rejected(tmp_path):
    events = records(tmp_path / "events")
    expected = records(tmp_path / "predicted", expectation(WIDTH, WIDTH + 1))
    with pytest.raises(ValueError, match="fixed by the start"):
        list(aggregate(events, 0, 2 * WIDTH, "sensor", "entity", expectations=expected))
    events = records(tmp_path / "events", completion(WIDTH))
    with pytest.raises(ValueError, match="outside the confirmed"):
        list(aggregate(events, 0, WIDTH, "sensor", "entity"))
    events = records(tmp_path / "events", completion(0, "/private?secret"))
    with pytest.raises(ValueError, match="route identifier"):
        list(aggregate(events, 0, WIDTH, "sensor", "entity"))

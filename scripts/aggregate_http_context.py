"""Convert normalized, covered HTTP access logs to NetWatch context files."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile


WIDTH = 5_000_000_000
MAX_I64 = (1 << 63) - 1
ROUTE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
SOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")
METHODS = {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}


def count(value):
    if type(value) is not int or not 0 <= value <= MAX_I64:
        raise ValueError("expected a nonnegative signed 64-bit integer")
    return value


def timestamp(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,19}", value):
        value = int(value)
    return count(value)


def route(row):
    name, method = row["route_id"], row["method"]
    if not isinstance(name, str) or not ROUTE.fullmatch(name) or not isinstance(method, str) or method not in METHODS:
        raise ValueError("expected a route identifier and a standard HTTP method")
    return name, method


def rows(path):
    with path.open() as stream:
        for lineno, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                yield row
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


@dataclass
class Operation:
    completed: int = 0
    request_bytes: int | None = 0
    response_bytes: int | None = 0
    expected: int | None = None

    def observe(self, row):
        self.completed += 1
        self.request_bytes = self.add_bytes(self.request_bytes, row.get("request_body_bytes"))
        self.response_bytes = self.add_bytes(self.response_bytes, row.get("response_body_bytes"))

    @staticmethod
    def add_bytes(total, value):
        if value is not None:
            count(value)
        return None if total is None or value is None else count(total + value)

    def export(self, key):
        name, method = key
        return dict(route_id=name, method=method, completed_requests=self.completed,
                    request_body_bytes=self.request_bytes if self.completed else None,
                    response_body_bytes=self.response_bytes if self.completed else None,
                    expected_requests=self.expected)


@dataclass
class Health:
    completed: int = 0
    errors: int = 0
    mean: float = 0.0

    def observe(self, row):
        status = count(row["status"])
        latency = row["latency_ms"]
        if not 100 <= status <= 599 or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid completed transaction status or latency")
        self.completed += 1
        self.errors += status >= 500
        self.mean += (latency - self.mean) / self.completed


def window_schedule(planned, start, end):
    if planned is None:
        return {}
    intervals = json.loads(planned.read_text())
    if not isinstance(intervals, list):
        raise ValueError("expected a list of planned intervals")
    covered = set()
    previous = start
    for interval in intervals:
        first, last = timestamp(interval["start_ns"]), timestamp(interval["end_ns"])
        if not previous <= first < last <= end or first % WIDTH or last % WIDTH:
            raise ValueError("planned intervals must be ordered, aligned and inside coverage")
        covered.update(range(first, last, WIDTH))
        previous = last
    return {stamp: stamp in covered for stamp in range(start, end, WIDTH)}


def aggregate(events, start, end, sensor, entity, *, expectations=None, planned=None, source=None):
    if not 0 <= start < end <= MAX_I64 or start % WIDTH or end % WIDTH:
        raise ValueError("coverage must be an aligned, nonempty interval")
    if source and not SOURCE.fullmatch(f"{source}.{end}"):
        raise ValueError("invalid source reference prefix")
    schedule = window_schedule(planned, start, end)
    operations = defaultdict(dict)
    health = defaultdict(Health)
    for row in rows(events):
        completion = timestamp(row["completed_at_ns"])
        if not start <= completion < end:
            raise ValueError("completed transaction outside the confirmed observation interval")
        stamp = completion // WIDTH * WIDTH
        health[stamp].observe(row)
        operations[stamp].setdefault(route(row), Operation()).observe(row)

    if expectations is not None:
        for row in rows(expectations):
            stamp, as_of = timestamp(row["window_start_ns"]), timestamp(row["as_of_ns"])
            if not start <= stamp < end or stamp % WIDTH or as_of > stamp:
                raise ValueError("demand estimate must be fixed by the start of its covered window")
            op = operations[stamp].setdefault(route(row), Operation())
            if op.expected is not None:
                raise ValueError("duplicate demand estimate")
            op.expected = count(row["expected_requests"])

    for stamp, by_route in operations.items():
        if len(by_route) > 32:
            raise ValueError(f"more than 32 tracked route/method pairs for window {stamp}")

    for stamp in range(start, end, WIDTH):
        sample = health.get(stamp, Health())
        record = dict(schema_version=1, sensor_id=sensor, entity_id=entity,
                      window_start_ns=str(stamp), window_end_ns=str(stamp + WIDTH),
                      completed_requests=sample.completed, server_errors=sample.errors,
                      mean_latency_ms=sample.mean if sample.completed else None,
                      planned_activity=schedule.get(stamp))
        if by_route := operations.get(stamp):
            record["operations"] = [value.export(key) for key, value in sorted(by_route.items())]
        if source:
            record["source_ref"] = f"{source}.{stamp}"
        yield record


def publish(folder, record, group_id=None):
    path = folder / f"{record['window_start_ns']}.json"
    content = (json.dumps(record, allow_nan=False) + "\n").encode()
    stream = tempfile.NamedTemporaryFile(dir=folder, prefix=".http-context-", delete=False)
    try:
        pending = Path(stream.name)
        with stream:
            fd = stream.fileno()
            if group_id is not None and os.fstat(fd).st_gid != group_id:
                os.fchown(fd, -1, group_id)
            os.fchmod(fd, 0o640 if group_id is not None else 0o600)
            stream.write(content)
        try:
            os.link(pending, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"refusing to change previously exported measurement: {path}") from None
            if group_id is not None:
                existing = path.stat()
                if existing.st_gid != group_id or stat.S_IMODE(existing.st_mode) != 0o640:
                    raise ValueError(f"existing measurement permissions differ from the requested group: {path}") from None
    finally:
        pending.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--planned", type=Path)
    parser.add_argument("--expectations", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--group-id", type=int, help="publish owner/group-readable files with this numeric GID")
    args = parser.parse_args()
    if args.group_id is not None and not 0 <= args.group_id < (1 << 32) - 1:
        parser.error("--group-id must be a valid nonnegative numeric GID")
    args.output.mkdir(exist_ok=True, parents=True)
    for record in aggregate(args.events, args.start_ns, args.end_ns, args.sensor, args.entity,
                            planned=args.planned, expectations=args.expectations, source=args.source):
        publish(args.output, record, args.group_id)


if __name__ == "__main__":
    main()

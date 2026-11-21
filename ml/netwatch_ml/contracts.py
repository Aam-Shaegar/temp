from functools import cache
import json
import ipaddress
from pathlib import Path
import re
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
ROUTE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
SOURCE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")

@cache
def validator(kind: str):
    if kind not in {"window", "prediction", "packet"}:
        raise ValueError("unknown contract")
    schema = json.loads((ROOT / "contracts" / f"{kind}.schema.json").read_text())
    return Draft202012Validator(schema)


def validate(kind: str, value: dict):
    validator(kind).validate(value)
    # jsonschema accepts Python NaN as number; reject it before inference.
    json.dumps(value, allow_nan=False)
    for key in ("window_start_ns", "window_end_ns", "timestamp_ns"):
        if key in value and not 0 <= int(value[key]) <= 2**63-1:
            raise ValueError("timestamp outside int64")
    if kind == "packet" and value["decode_status"] == "decoded":
        ipaddress.ip_address(value["src_ip"])
        ipaddress.ip_address(value["dst_ip"])
        if value["protocol"] in {"TCP","UDP"} and (value["src_port"] is None or value["dst_port"] is None):
            raise ValueError("missing transport port")
    if kind == "window":
        if int(value["window_end_ns"]) <= int(value["window_start_ns"]):
            raise ValueError("window end must exceed start")
        if app := value.get("application"):
            start, end = int(value["window_start_ns"]), int(value["window_end_ns"])
            count, errors = app["completed_requests"], app["server_errors"]
            if (type(count) is not int or type(errors) is not int or errors > count
                    or (count == 0) != (app["mean_latency_ms"] is None)
                    or value["entity_type"] != "destination_service" or not value["entity_id"].startswith("TCP/")
                    or end - start != 5_000_000_000 or start % 5_000_000_000):
                raise ValueError("invalid application context")
            validate_operation_counts(app)
        for key, number in value["features"].items():
            if number is not None and (key.endswith(("packets", "bytes", "flows", "attempts")) or key.startswith("unique_") or key=="completed_flow_count"):
                if isinstance(number,bool) or not isinstance(number,int) or number>2**63-1:
                    raise ValueError("count must be int64")
    if kind == "prediction":
        scored = value["status"] in {"normal", "anomalous"}
        if scored and (value["score"] is None or value["threshold"] is None):
            raise ValueError("scored status needs numbers")
        if not scored and (value["score"] is not None or value["threshold"] is not None):
            raise ValueError("unavailable prediction must not have a score")
    return value


def validate_operation_counts(app):
    if "source_ref" in app and not SOURCE_REF.fullmatch(app["source_ref"]):
        raise ValueError("invalid application source reference")
    remaining = app["completed_requests"]
    seen = set()
    for operation in app.get("operations", ()):
        route, method = operation["route_id"], operation["method"]
        key = (route, method)
        completed = operation["completed_requests"]
        if not ROUTE_ID.fullmatch(route) or key in seen:
            raise ValueError("invalid or duplicate application route and method")
        seen.add(key)
        if type(completed) is not int or completed > remaining:
            raise ValueError("application operation counts exceed completed requests")
        remaining -= completed
        for field in ("request_body_bytes", "response_body_bytes", "expected_requests"):
            value = operation[field]
            if value is not None and (type(value) is not int or field != "expected_requests" and completed == 0 and value != 0):
                raise ValueError(f"invalid application operation {field}")

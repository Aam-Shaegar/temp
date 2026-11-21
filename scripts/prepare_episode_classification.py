"""Apply the same Go extraction and annotation join to new isolated service traces."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

NS = 1_000_000_000
WIDTH = 5 * NS
FEATURE_SET = "traffic-v1"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root, sensor, destination):
    if destination.exists():
        raise FileExistsError(destination)
    dataset = []
    for manifest in sorted(root.glob("*/manifest.json")):
        info = json.loads(manifest.read_text())
        capture = manifest.with_name("traffic.pcap")
        group = manifest.parent.name
        source_sha256 = sha256(capture)
        if source_sha256 != info["sha256"] or group != f"run-{info['seed']}":
            raise ValueError(f"{group}: capture provenance mismatch")
        if "0 packets dropped by kernel" not in info["capture_log"].splitlines():
            raise ValueError(f"{group}: capture loss")
        phases = info["intervals"]
        if (not phases or any(p["start"] >= p["end"] or p["requests"] <= 0 for p in phases)
                or any(a["end"] > b["start"] for a, b in zip(phases, phases[1:]))):
            raise ValueError(f"{group}: incompatible workload timeline")
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "windows.jsonl"
            result = subprocess.run(
                [str(sensor), "replay", "--input", str(capture), "--output", str(output),
                 "--service", "127.0.0.1:8080", "--epoch", group, "--feature-set", FEATURE_SET,
                 "--coverage-start", str(phases[0]["start"]),
                 "--coverage-end", str(phases[-1]["end"] + 900_000_000)],
                check=True, capture_output=True, text=True,
            )
            stats = json.loads(result.stderr)
            if stats["late_packets"] != 0 or stats["status_counts"] != {"decoded": stats["records"]}:
                raise ValueError(f"{group}: invalid replay: {stats}")
            windows = [json.loads(line) for line in output.read_text().splitlines()]
        for index, phase in enumerate(phases):
            enclosed = [
                w for w in windows if phase["start"] <= int(w["window_start_ns"])
                < int(w["window_end_ns"]) <= phase["end"]
            ]
            expected_windows = phase["end"] // WIDTH - (phase["start"] + WIDTH - 1) // WIDTH
            if len(enclosed) != expected_windows:
                raise ValueError(f"{group}: missing windows in phase {index}")
            for w in enclosed:
                if w["quality"]["capture_dropped"] is not None:
                    raise ValueError(f"{group}: unexpected archived drop counter")
                w["quality"]["capture_dropped"] = 0  # Verified against the recorder above.
                dataset.append(dict(
                    group=group, phase=index, label="normal" if phase["label"] == "benign" else phase["label"],
                    start_ns=int(w["window_start_ns"]), end_ns=int(w["window_end_ns"]),
                    feature_set_version=w["feature_set_version"], features=w["features"],
                    quality=w["quality"], source_sha256=source_sha256, entity_type=w["entity_type"],
                ))
    if not dataset:
        raise ValueError("no completed captures")
    with destination.open("x") as out:
        for row in dataset:
            out.write(json.dumps(row, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "sensor", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.root, args.sensor.resolve(strict=True), args.destination)

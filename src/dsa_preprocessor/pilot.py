"""Bounded, reproducible comparison jobs with an explicit pair-fit budget."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import platform
import tempfile
import time

from .pipeline import _code_provenance, _dataset_root, _entry, _registration_paths, prepare


PURPOSES = {"development", "confirmation", "reverse", "ablation", "retry", "smoke", "identity"}


def _save(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _validate_packet(packet, base, max_fits):
    if not isinstance(packet, dict) or packet.get("schema_version") != 1:
        raise ValueError("Pilot packet must use schema_version 1")
    cap = max_fits if max_fits is not None else packet.get("max_fits", 160)
    if type(cap) is not int or not 1 <= cap <= 160:
        raise ValueError("Pilot max_fits must be an integer in 1..160")
    jobs = packet.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Pilot jobs must be a nonempty list")
    seen = set()
    result = []
    total = 0
    for item in jobs:
        if not isinstance(item, dict) or set(item)-{"job_id", "source_file", "config", "purpose", "notes", "retry_of"}:
            raise ValueError("Pilot job contains unsupported fields")
        name = item.get("job_id")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name)
                or name in seen):
            raise ValueError("Pilot job_id must be a unique safe directory name")
        seen.add(name)
        if item.get("purpose") not in PURPOSES:
            raise ValueError("Every pilot job requires an explicit comparison purpose")
        if not isinstance(item.get("config"), dict):
            raise ValueError("Every pilot job requires an explicit configuration")
        raw_source = item.get("source_file")
        if not isinstance(raw_source, str) or not raw_source:
            raise ValueError("Every pilot job requires a source_file")
        path = Path(raw_source).expanduser()
        source = (base/path).resolve() if not path.is_absolute() else path.resolve()
        entry = _entry(source)
        config = item["config"]
        indices = config.get("frames")
        if indices is None:
            indices = [i for i in range(entry["frames"]) if i != entry["summary_frame"]]
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or not 0 <= index < entry["frames"] for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Pilot jobs need valid unique original frame indices")
        if config.get("reference_frame") not in indices:
            raise ValueError("Pilot jobs require an explicit selected reference_frame")
        if entry["summary_frame"] in indices:
            raise ValueError("Scanner summary images cannot be pilot motion-estimation frames")
        # Reserve distinct directed edges, including duplicate/uninformative attempts.
        registration = config.get("registration")
        edges = None
        if registration is not None:
            if not isinstance(registration, dict):
                raise ValueError("registration must be an object")
            paths, routes = _registration_paths(registration.get("paths", {}), indices, config["reference_frame"])
            check = registration.get("check_paths_directly", False)
            if type(check) is not bool:
                raise ValueError("registration.check_paths_directly must be boolean")
            if registration.get("fitting_landmarks") and any(paths.values()):
                raise ValueError("Reference-frame fitting landmarks cannot be mixed with keyframe paths")
            edges = {edge for route in routes.values() for edge in zip(route[:-1], route[1:])}
            if check:
                edges |= {(config["reference_frame"], target) for target, route in routes.items() if len(route) > 2}
        reserved = len(edges) if edges is not None else len(indices)-1
        total += reserved
        result.append({**item, "source_file": str(source), "reserved_pair_fits": reserved,
                       "reserved_directed_edges": [list(edge) for edge in sorted(edges)] if edges is not None else None})
    if total > cap:
        raise ValueError(f"Pilot reserves {total} pair fits, exceeding the cap of {cap}; no jobs were run")
    return result, cap, total


def run_pilot(packet_file, output_dir, *, max_fits=None):
    """Execute a predeclared matrix; all reverse/retry/ablation jobs consume budget.

    A numerical inverse is not a reverse registration. This utility does not tune
    automatically or select a winner from image similarity.
    """
    packet_path = Path(packet_file).expanduser().resolve()
    raw_packet = packet_path.read_bytes()
    packet = json.loads(raw_packet)
    jobs, cap, total = _validate_packet(packet, packet_path.parent, max_fits)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Pilot output exists; choose a new directory")
    for job in jobs:
        dataset = _dataset_root(job["source_file"])
        if dataset and output.is_relative_to(dataset):
            raise ValueError("Write pilot derivatives outside the source dataset")
    output.mkdir(parents=True)
    (output/"packet.json").write_text(json.dumps(packet, indent=2, allow_nan=False)+"\n")
    record = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "packet_file": str(packet_path), "packet_sha256": hashlib.sha256(raw_packet).hexdigest(),
              "max_fits": cap, "reserved_pair_fits": total, "attempted_pair_fits": 0,
              "completed_pair_fits": 0, "consumed_pair_budget": 0,
              "fit_counts_exact": True, "jobs": [], "status": "running",
              "anatomical_acceptance_established": False, "provenance": _code_provenance(),
              "budget_note": "Every predeclared source/reference attempt, including reverse fits and retries, consumes the reservation; numerical inverses are not independent reverse fits."}
    _save(output/"pilot_summary.json", record)
    for job in jobs:
        start = time.perf_counter()
        record["consumed_pair_budget"] += job["reserved_pair_fits"]
        row = {"job_id": job["job_id"], "purpose": job["purpose"],
               "source_file": job["source_file"], "reserved_pair_fits": job["reserved_pair_fits"],
               "reserved_directed_edges": job["reserved_directed_edges"],
               "output_dir": str(output/job["job_id"]), "config": job["config"]}
        try:
            manifest = prepare(job["source_file"], output/job["job_id"], job["config"])
            row.update(status="prepared", source_sha256=manifest["source_sha256"],
                       registration_fits=manifest.get("registration_fits"),
                       registration_completed_fits=manifest.get("registration_completed_fits", manifest.get("registration_fits")),
                       fit_counts_exact="registration_fits" in manifest,
                       registration_method=manifest.get("registration_method", "legacy_translation"),
                       candidate_frames=sum(frame.get("registration_valid", frame["status"] in {"reference", "estimated", "manual"})
                                            for frame in manifest["frames"]),
                       annotation_eligible_frames=sum(frame.get("annotation_eligible", False) for frame in manifest["frames"]))
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                       registration_fits=getattr(exc, "registration_attempted_fits", None),
                       registration_completed_fits=getattr(exc, "registration_completed_fits", None),
                       fit_counts_exact=hasattr(exc, "registration_attempted_fits"))
        # The budget stays conservative; actual calls remain explicitly unknown when unavailable.
        if row["fit_counts_exact"] and record["fit_counts_exact"]:
            record["attempted_pair_fits"] += row["registration_fits"]
            record["completed_pair_fits"] += row["registration_completed_fits"]
        else:
            record["fit_counts_exact"] = False
            record["attempted_pair_fits"] = None
            record["completed_pair_fits"] = None
        if row["registration_fits"] is not None and row["registration_fits"] > job["reserved_pair_fits"]:
            raise RuntimeError("Estimator calls exceeded the predeclared directed-edge reservation")
        row["elapsed_seconds"] = time.perf_counter()-start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row["process_peak_rss_bytes"] = int(peak if platform.system() == "Darwin" else peak*1024)
        row["peak_rss_scope"] = "Cumulative process high-water mark; not an isolated per-job allocation"
        record["jobs"].append(row)
        _save(output/"pilot_summary.json", record)
    record["status"] = "completed_with_failures" if any(row["status"] == "failed" for row in record["jobs"]) else "completed"
    record["elapsed_seconds"] = sum(row["elapsed_seconds"] for row in record["jobs"])
    _save(output/"pilot_summary.json", record)
    return record

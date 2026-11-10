"""Command line entry points for the DSA preparation workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


def _frame_list(value: str) -> list[int]:
    """Parse explicit zero-based indices and inclusive ranges, e.g. 2,4-8."""
    frames = []
    try:
        for part in value.split(","):
            if "-" in part:
                first, last = map(int, part.split("-"))
                if first < 0 or last < first:
                    raise ValueError
                frames.extend(range(first, last + 1))
            else:
                index = int(part)
                if index < 0:
                    raise ValueError
                frames.append(index)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Use zero-based indices such as 1,3-7,9") from None
    if not frames or len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("Frame indices must be nonempty and unique")
    return sorted(frames)


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return result


def _write_json_new(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _print_inspection(entries: list[dict]) -> None:
    print("run\tframes\twidth\theight\tsummary_frame\tpixel_spacing_known")
    for item in entries:
        print("\t".join(str(item.get(key, "")) for key in (
            "run_id", "frames", "width", "height", "summary_frame", "pixel_spacing_known")))
    print(f"{len(entries)} sequence(s)", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsa-preprocess",
        description="Align 2D DSA frames for one fixed segmentation in 3D Slicer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="List NRRD runs without modifying the dataset")
    inspect.add_argument("dataset", type=Path)
    inspect.add_argument("--json", action="store_true", help="Print complete inventory as JSON")
    inspect.add_argument("--subjects", nargs="+", help="Only these subject folders, e.g. sub-example01 sub-example02")

    config = sub.add_parser("configure", help="Write an editable starting configuration")
    config.add_argument("input", type=Path)
    config.add_argument("-o", "--output", required=True, type=Path)

    prepare = sub.add_parser("prepare", help="Register one NRRD run and create Slicer + review files")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("-o", "--output", required=True, type=Path)
    prepare.add_argument("--config", type=Path)
    prepare.add_argument("--reference-frame", type=int, help="Original zero-based frame index")
    prepare.add_argument("--frames", type=_frame_list, help="Original indices, e.g. 2,4-10")
    prepare.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    prepare.add_argument("--max-shift-px", type=float)
    prepare.add_argument("--polarity", choices=("dark", "bright"))
    prepare.add_argument("--registration-method", choices=("translation", "rigid", "similarity", "affine", "bspline", "tvl1", "demons"),
                         help="Opt in to schema-2 vessel registration; all results require anatomical review")

    evaluate = sub.add_parser("evaluate", help="Measure independent vessel landmarks in reference coordinates")
    evaluate.add_argument("prepared", type=Path)
    evaluate.add_argument("--landmarks", type=Path)
    evaluate.add_argument("--template", action="store_true", help="Create an empty fingerprinted landmark collection packet")
    evaluate.add_argument("-o", "--output", required=True, type=Path)

    pilot = sub.add_parser("pilot", help="Run a fixed comparison packet with a pair-fit budget")
    pilot.add_argument("packet", type=Path)
    pilot.add_argument("-o", "--output", required=True, type=Path)
    pilot.add_argument("--max-fits", type=int, help="Override packet cap; must be at most 160, including reverse/retry jobs")

    batch = sub.add_parser("batch", help="Prepare dataset runs separately and record failures")
    batch.add_argument("dataset", type=Path)
    batch.add_argument("-o", "--output", required=True, type=Path)
    batch.add_argument("--config-dir", type=Path, help="Optional <run_id>.json configurations")
    batch.add_argument("--subjects", nargs="+", help="Only these subject folders")
    batch.add_argument("--limit", type=int, help="Process only the first N matching runs")
    return parser


def _filter_subjects(entries: list[dict], subjects: list[str] | None) -> list[dict]:
    if not subjects:
        return entries
    requested = set(subjects)
    def subject(item):
        return str(item["run_id"]).split("_")[0]
    available = {subject(item) for item in entries}
    missing = requested - available
    if missing:
        raise ValueError(f"No sequences found for: {', '.join(sorted(missing))}")
    return [item for item in entries if subject(item) in requested]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from . import pipeline
    try:
        if args.command == "inspect":
            entries = _filter_subjects(pipeline.inspect_dataset(args.dataset), args.subjects)
            if args.json:
                print(json.dumps(entries, indent=2, allow_nan=False))
            else:
                _print_inspection(entries)
            return 0
        if args.command == "configure":
            config = pipeline.suggest_config(args.input)
            _write_json_new(args.output, config)
            print(args.output.resolve())
            return 0
        if args.command == "prepare":
            config = _load_config(args.config)
            for field in ("reference_frame", "frames", "roi", "max_shift_px", "polarity"):
                value = getattr(args, field)
                if value is not None:
                    config[field] = value
            if args.registration_method is not None:
                nested = config.get("registration", {})
                if not isinstance(nested, dict):
                    raise ValueError("registration must be an object")
                config["registration"] = {**nested, "method": args.registration_method}
            manifest = pipeline.prepare(args.input, args.output, config)
            flagged = sum(frame.get("status") not in ("accepted", "reference", "manual", "ok", "estimated")
                          for frame in manifest["frames"])
            if manifest.get("schema_version") == 2:
                candidates = sum(frame.get("registration_valid", False) for frame in manifest["frames"])
                print(f"Prepared {len(manifest['frames'])} frame(s); {candidates} numerically supported candidates. Anatomical review is required.")
            else:
                print(f"Prepared {len(manifest['frames'])} frame(s); {flagged} automatically flagged. All alignments need visual review.")
            print(f"Review: {(args.output / 'report.html').resolve()}")
            return 0
        if args.command == "evaluate":
            from .evaluation import annotation_template, evaluate_landmarks
            if args.template == (args.landmarks is not None):
                raise ValueError("Choose either --template or --landmarks FILE")
            result = annotation_template(args.prepared) if args.template else evaluate_landmarks(args.prepared, args.landmarks)
            prepared_manifest = json.loads((args.prepared / "manifest.json").read_text())
            source = prepared_manifest.get("source_file")
            dataset = pipeline._dataset_root(source) if source else None
            if dataset and args.output.expanduser().resolve().is_relative_to(dataset):
                raise ValueError("Write evaluation derivatives outside the source dataset")
            _write_json_new(args.output, result)
            print(args.output.resolve())
            if not args.template:
                print(f"{result['status']}; anatomical acceptance remains subject to independent review")
            return 0
        if args.command == "pilot":
            from .pilot import run_pilot
            result = run_pilot(args.packet, args.output, max_fits=args.max_fits)
            print(f"{result['status']}; reserved {result['reserved_pair_fits']}/{result['max_fits']} pair fits")
            print((args.output / "pilot_summary.json").resolve())
            return 1 if result["status"] == "completed_with_failures" else 0
        if args.command == "batch":
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be a positive integer")
            entries = _filter_subjects(pipeline.inspect_dataset(args.dataset), args.subjects)
            if args.limit is not None:
                entries = entries[:args.limit]
            if not entries:
                raise ValueError("No matching NRRD sequences found")
            dataset_path = args.dataset.expanduser().resolve()
            if dataset_path.is_dir() and args.output.expanduser().resolve().is_relative_to(dataset_path):
                raise ValueError("Write preprocessing derivatives outside the source dataset")
            # Refuse to reuse a batch directory: prevents mixing stale and current results.
            args.output.mkdir(parents=True, exist_ok=False)
            results = []
            for number, entry in enumerate(entries, 1):
                run = entry["run_id"]
                print(f"[{number}/{len(entries)}] {run}", flush=True)
                config_path = args.config_dir / f"{run}.json" if args.config_dir else None
                try:
                    config = _load_config(config_path if config_path and config_path.exists() else None)
                    manifest = pipeline.prepare(entry["source_file"], args.output / run, config)
                    results.append({"run_id": run, "status": "prepared", "frames": len(manifest["frames"])})
                except Exception as exc:
                    # A broken run must not prevent processing the other acquisitions.
                    results.append({"run_id": run, "status": "failed", "error": str(exc)})
                    print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
                _save_batch_summary(args.output / "batch_summary.json", results)
            failed = sum(row["status"] == "failed" for row in results)
            print(f"Prepared {len(results) - failed}; failed {failed}. Summary: {args.output / 'batch_summary.json'}")
            return 1 if failed else 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _save_batch_summary(path: Path, results: list[dict]) -> None:
    # Atomic updates allow inspecting progress while a larger batch runs.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

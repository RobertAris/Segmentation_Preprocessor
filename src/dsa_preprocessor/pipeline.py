"""Dataset discovery, explicit frame selection, provenance, and Slicer preparation."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import nrrd
import numpy as np

from . import __version__
from .io import read_sequence, spatial_header, write_sequence, write_volume
from .registration import apply_shift, correlation, features, register_frames, roi_slices, valid_minip


DEFAULTS = {"reference_frame": None, "frames": None, "roi": None, "max_shift_px": 32.,
            "upsample_factor": 10, "normalization": "none", "polarity": "dark",
            "min_correlation": .35, "manual_shifts": {}, "landmarks": {},
            "exclude_from_minip": [], "allow_summary_frame": False}


def _dataset_root(path):
    for parent in [Path(path).resolve().parent, *Path(path).resolve().parents]:
        if (parent / "series.tsv").is_file():
            return parent
    return None


def _metadata(path):
    sidecar = Path(path).with_suffix(".json")
    return json.loads(sidecar.read_text()) if sidecar.is_file() else {}


def _series_notes(path):
    root = _dataset_root(path)
    if root is None:
        return ""
    with (root / "series.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            run = f"{row['participant_id']}_acq-{row['acq']}_run-{row['run']}_{row['suffix']}"
            if run == Path(path).stem:
                return row.get("note", "")
    return ""


def _entry(path):
    path = Path(path).resolve()
    header = nrrd.read_header(str(path))
    sizes = list(map(int, header["sizes"]))
    if len(sizes) == 3:
        w, h, count = sizes
    elif len(sizes) == 4 and sizes[2] == 1:
        w, h, _, count = sizes
    else:
        raise ValueError(f"Not a supported 2D+time NRRD: {path.name}")
    meta, note = _metadata(path), _series_notes(path)
    summary = count-1 if "last frame" in note.lower() and "summary" in note.lower() else None
    return {"source_file": str(path), "run_id": path.stem.removesuffix("_angio"),
            "frames": count, "width": w, "height": h, "summary_frame": summary,
            "pixel_spacing_known": meta.get("PixelSpacingKnown"),
            "frame_order_certain": meta.get("FrameOrderCertain"), "note": note}


def inspect_dataset(root):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Input does not exist: {root}")
    paths = [root] if root.is_file() else sorted(root.rglob("*_angio.nrrd"))
    return [_entry(path) for path in paths]


def _frame_scores(frames, polarity="dark"):
    # Downsample only this selection heuristic; original frames remain untouched.
    step = max(1, max(frames.shape[1:])//512)
    result = []
    for frame in frames:
        f = features(frame[::step, ::step], polarity)
        roi = roi_slices(f.shape)
        f = f[roi]
        result.append(float(np.percentile(f, 95)))
    return result


def suggest_config(source_file):
    entry = _entry(source_file)
    frames, _ = read_sequence(source_file)
    selected = [i for i in range(len(frames)) if i != entry["summary_frame"]]
    if not selected:
        raise ValueError("No acquired frames remain after summary exclusion")
    scores = _frame_scores(frames)
    config = dict(DEFAULTS)
    config.update(frames=selected, reference_frame=max(selected, key=lambda i: scores[i]))
    return config


def _indices(value, count, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of original zero-based frame indices")
    if any(type(i) is not int or not 0 <= i < count for i in value) or len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique integers in 0..{count-1}")
    return sorted(value)


def _manual_override(index, config):
    manual = config["manual_shifts"].get(str(index))
    landmarks = config["landmarks"].get(str(index))
    if manual is not None and landmarks is not None:
        raise ValueError(f"Use either manual_shifts or landmarks for frame {index}, not both")
    if landmarks is not None:
        reference = np.asarray(landmarks["reference"], dtype=float)
        moving = np.asarray(landmarks["moving"], dtype=float)
        if reference.shape != moving.shape or reference.ndim != 2 or reference.shape[1] != 2 or not len(reference):
            raise ValueError("Landmarks require matching nonempty Nx2 arrays of [x,y] coordinates")
        if not np.isfinite(reference).all() or not np.isfinite(moving).all():
            raise ValueError("Landmarks must be finite")
        differences = reference-moving
        xy = np.median(differences, axis=0)
        residual = np.linalg.norm(differences-xy, axis=1)
        return xy[::-1], {"landmark_residual_rms_px": float(np.sqrt(np.mean(residual**2))),
                           "landmark_residual_max_px": float(residual.max()), "landmark_count": len(reference)}
    if manual is not None:
        manual = np.asarray(manual, dtype=float)
        if manual.shape != (2,) or not np.isfinite(manual).all():
            raise ValueError("manual_shifts values must be finite [dy,dx] pairs")
        return manual, {}
    return None, {}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_file, output_dir, config=None):
    source_file = Path(source_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output_dir}")
    dataset = _dataset_root(source_file)
    if output_dir == source_file or (dataset and output_dir.is_relative_to(dataset)):
        raise ValueError("Write preprocessing derivatives outside the source dataset")
    supplied = dict(config or {})
    unknown = set(supplied)-set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    config = {**DEFAULTS, **supplied}
    entry = _entry(source_file)
    originals_all, header = read_sequence(source_file)
    count = len(originals_all)
    if config["frames"] is None:
        selected = [i for i in range(count) if i != entry["summary_frame"]]
    else:
        selected = _indices(config["frames"], count, "frames")
    if not selected:
        raise ValueError("No acquired frames selected")
    if entry["summary_frame"] in selected and not config["allow_summary_frame"]:
        raise ValueError("Selection contains scanner summary frame; remove it or set allow_summary_frame=true")
    scores = _frame_scores(originals_all, config["polarity"])
    reference = config["reference_frame"]
    if reference is None:
        reference = max(selected, key=lambda i: scores[i])
    if type(reference) is not int or reference not in selected:
        raise ValueError("reference_frame must be an original frame index included in frames")
    for key in ("manual_shifts", "landmarks"):
        if not isinstance(config[key], dict):
            raise ValueError(f"{key} must be an object keyed by original frame indices")
        normalized = {str(index): value for index, value in config[key].items()}
        if len(normalized) != len(config[key]):
            raise ValueError(f"Duplicate original frame keys in {key}")
        config[key] = normalized
        if any(str(i) not in {str(s) for s in selected} for i in config[key]):
            raise ValueError(f"{key} contains a frame outside the selected sequence")
    excluded = config["exclude_from_minip"]
    if not isinstance(excluded, list) or any(type(i) is not int or i not in selected for i in excluded):
        raise ValueError("exclude_from_minip must list selected original frame indices")
    config.update(reference_frame=reference, frames=selected)
    originals = originals_all[selected]
    del originals_all
    shifts, metrics = register_frames(originals, {**config, "reference_index": selected.index(reference)})
    registration_roi = roi_slices(originals.shape[1:], config["roi"])
    ref_features = features(originals[selected.index(reference)], config["polarity"])[registration_roi]
    for pos, index in enumerate(selected):
        manual, detail = _manual_override(index, config)
        if manual is not None:
            if index == reference and np.any(manual != 0):
                raise ValueError("Reference-frame shift must remain zero")
            shifts[pos] = manual
            af, valid = apply_shift(features(originals[pos], config["polarity"])[registration_roi], manual)
            metrics[pos].update(status="manual" if index != reference else "reference",
                                ncc_after=correlation(ref_features, af, valid),
                                reason="User-specified translation; verify shared anatomy", **detail)
    aligned = np.empty(originals.shape, dtype=np.float32)
    validity = np.empty(originals.shape, dtype=bool)
    frame_info = []
    for pos, index in enumerate(selected):
        aligned[pos], validity[pos] = apply_shift(originals[pos], shifts[pos])
        info = {**metrics[pos], "source_frame": index, "shift_y_px": float(shifts[pos, 0]),
                "shift_x_px": float(shifts[pos, 1]), "selection_score": scores[index],
                "valid_fraction": float(validity[pos].mean())}
        info["included_in_minip"] = info["status"] in ("reference", "estimated", "manual") and index not in excluded
        frame_info.append(info)
    include = [info["included_in_minip"] for info in frame_info]
    # Bright-vessel data need a maximum projection, while retaining a common output filename.
    composite, composite_valid = valid_minip(aligned if config["polarity"] == "dark" else -aligned, validity, include)
    if config["polarity"] == "bright":
        composite = -composite
    warnings = ["Automatic registration and frame selection require visual verification; correlation is not anatomical accuracy."]
    if len(selected) == 1:
        warnings.append("Single-frame input: there is no temporal motion to correct.")
    if entry["pixel_spacing_known"] is not True:
        warnings.append("Pixel spacing is unknown or unverified; do not interpret placeholder spacing as measured geometry.")
    if entry["frame_order_certain"] is False:
        warnings.append("Source metadata marks temporal frame order uncertain; original stored indices are retained.")
    if not any(include):
        warnings.append("No frames passed the composite filter; the composite validity mask is empty.")
    manifest = {"schema_version": 1, "software_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": entry["run_id"],
                "source_file": str(source_file), "source_sha256": _sha256(source_file),
                "sequence_file": "registered.seq.nrrd", "reference_file": "reference.nrrd",
                "minip_file": "minip.nrrd", "validity_file": "validity.seq.nrrd",
                "minip_validity_file": "minip_validity.nrrd", "reference_frame": reference,
                "spatial_shape": list(originals.shape[1:]), "frames": frame_info, "config": config,
                "source_frame_count": count, "summary_frame": entry["summary_frame"],
                "pixel_spacing_known": entry["pixel_spacing_known"],
                "reference_selection": "automatic" if supplied.get("reference_frame") is None else "user",
                "review_required": True, "warnings": warnings,
                "projection": "minimum" if config["polarity"] == "dark" else "maximum",
                "transform_convention": "moving_to_reference: output[y,x] = source[y-dy,x-dx]; pixels; no wrapping",
                "geometry": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
                             for key, value in spatial_header(header).items() if key != "encoding"},
                "dependencies": {package: version(package) for package in ("numpy", "scipy", "scikit-image", "pynrrd")}}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        write_sequence(temporary / manifest["sequence_file"], aligned, header, selected)
        write_sequence(temporary / manifest["validity_file"], validity.astype(np.uint8), header, selected)
        write_volume(temporary / manifest["reference_file"], originals[selected.index(reference)], header)
        write_volume(temporary / manifest["minip_file"], composite, header)
        write_volume(temporary / manifest["minip_validity_file"], composite_valid.astype(np.uint8), header)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        (temporary / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
        with (temporary / "transforms.tsv").open("w", newline="") as stream:
            fields = ["source_frame", "shift_y_px", "shift_x_px", "status", "ncc_before", "ncc_after", "included_in_minip", "reason"]
            writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(frame_info)
        from .report import write_report
        write_report(temporary, manifest, originals, aligned, validity)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest

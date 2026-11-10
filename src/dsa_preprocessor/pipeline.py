"""Dataset discovery, explicit frame selection, provenance, and Slicer preparation."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import nrrd
import numpy as np
from scipy import ndimage

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
    unknown = set(supplied)-set(DEFAULTS)-{"registration"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    registration = supplied.get("registration")
    if "registration" in supplied and not isinstance(registration, dict):
        raise ValueError("registration must be an object; omit it for legacy translation")
    config = {**DEFAULTS, **supplied}
    if registration is not None:
        if config["manual_shifts"] or config["landmarks"]:
            raise ValueError("Do not mix registration with legacy manual_shifts/landmarks; use registration.fitting_landmarks")
        for key in ("max_shift_px", "upsample_factor", "normalization", "min_correlation"):
            if key in supplied and supplied[key] != DEFAULTS[key]:
                raise ValueError(f"{key} is a legacy translation option; configure the nested registration object instead")
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
    if registration is not None:
        return _prepare_fields(source_file, output_dir, config, supplied, entry,
                               originals, header, selected, reference, scores, count)
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
                "registration_fits": sum(info["source_frame"] != reference and info["ncc_before"] is not None for info in frame_info),
                "registration_completed_fits": sum(info["source_frame"] != reference and info["ncc_before"] is not None for info in frame_info),
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
        from .projections import write_projection_outputs
        manifest["projection_comparison"] = write_projection_outputs(
            temporary, manifest, originals, aligned, validity, composite, composite_valid, header)
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


def _json_values(value):
    """Convert numerical metadata to strict JSON while keeping missing QA explicit."""
    if isinstance(value, dict):
        return {str(key): _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _code_provenance():
    package = Path(__file__).resolve().parent
    try:
        root = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=package, text=True,
            stderr=subprocess.DEVNULL).strip()).resolve()
        # A wheel inside another project's .venv must not inherit that project's
        # Git revision. Only this package's tracked source checkout is eligible.
        revision = None
        if package == root / "src" / "dsa_preprocessor":
            subprocess.check_output(
                ["git", "ls-files", "--error-unmatch", "src/dsa_preprocessor/__init__.py", "pyproject.toml"],
                cwd=root, stderr=subprocess.DEVNULL)
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True,
                stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"code_revision": revision, "software_version": __version__,
            "source_code_sha256": {path.name: _sha256(path) for path in sorted(package.glob("*.py"))},
            "python": platform.python_version(), "platform": platform.platform()}


def _fitting_landmarks(value, selected, reference, shape):
    if not isinstance(value, dict):
        raise ValueError("registration.fitting_landmarks must be keyed by original frame indices")
    normalized = {str(key): points for key, points in value.items()}
    if len(normalized) != len(value):
        raise ValueError("Duplicate frame keys in registration.fitting_landmarks")
    for key, points in normalized.items():
        if key not in {str(index) for index in selected} or key == str(reference):
            raise ValueError("fitting_landmarks must target selected nonreference source frames")
        if not isinstance(points, dict) or set(points) != {"reference", "moving"}:
            raise ValueError("fitting_landmarks require reference and moving Nx2 [x,y] arrays")
        fixed, moving = (np.asarray(points[name], dtype=float) for name in ("reference", "moving"))
        if (fixed.ndim != 2 or fixed.shape[1:] != (2,) or fixed.shape != moving.shape
                or len(fixed) < 3 or not np.isfinite(fixed).all() or not np.isfinite(moving).all()):
            raise ValueError("fitting_landmarks require matching finite Nx2 arrays, N>=3")
        for coords in (fixed, moving):
            if ((coords < 0).any() or (coords[:, 0] > shape[1]-1).any()
                    or (coords[:, 1] > shape[0]-1).any()):
                raise ValueError("fitting_landmarks must lie inside original full-resolution frames")
    return normalized


def _registration_paths(value, selected, reference):
    """Validate explicit coordinate routes; their order does not imply timing."""
    if not isinstance(value, dict):
        raise ValueError("registration.paths must be an object keyed by original target frames")
    normalized = {str(key): path for key, path in value.items()}
    if len(normalized) != len(value):
        raise ValueError("Duplicate target frame keys in registration.paths")
    available = set(selected)
    for key, path in normalized.items():
        if key not in {str(index) for index in selected if index != reference}:
            raise ValueError("registration.paths must target selected nonreference original frames")
        if (not isinstance(path, list) or any(type(index) is not int or index not in available for index in path)
                or len(set(path)) != len(path) or reference in path or int(key) in path):
            raise ValueError("Path intermediates must be unique selected original frames, excluding reference and target")
    routes = {index: [reference, *normalized.get(str(index), []), index]
              for index in selected if index != reference}
    return normalized, routes


def _sampling_support(field, shape):
    coords = np.indices(field.shape[1:], dtype=float) + field
    return ((coords[0] >= 0) & (coords[0] <= shape[0]-1)
            & (coords[1] >= 0) & (coords[1] <= shape[1]-1))


def _sample_through_pull(array, field):
    return ndimage.map_coordinates(np.asarray(array, dtype=float),
                                   np.indices(field.shape[1:], dtype=float)+field,
                                   order=1, mode="constant", cval=0., prefilter=False)


class _RegistrationGraph:
    """Cache directed fits on disk and compose explicit reference-anchored paths."""

    def __init__(self, originals, selected, reference, config, polarity, roi,
                 fitting, routes, direct_check, max_disagreement, temporary,
                 source_hash, geometry, estimate):
        self.images = {index: originals[pos] for pos, index in enumerate(selected)}
        self.reference = reference
        self.config, self.polarity, self.roi = config, polarity, roi
        self.fitting, self.routes = fitting, routes
        self.direct_check, self.max_disagreement = direct_check, max_disagreement
        self.directory, self.source_hash, self.geometry = temporary, source_hash, geometry
        self.estimate = estimate
        self.config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.edges = {}
        self.attempted_fits = self.completed_fits = 0
        self.shape = originals.shape[1:]

    def _load_edge(self, record):
        from .transforms import load_pull_field
        field_path = self.directory/record["field_file"]
        support_path = self.directory/record["support_file"]
        if _sha256(field_path) != record["field_sha256"] or _sha256(support_path) != record["support_sha256"]:
            raise ValueError("Cached registration edge changed during preparation")
        field, _ = load_pull_field(field_path)
        with np.load(support_path, allow_pickle=False) as data:
            source_support = data["source_support"].copy()
            correspondence = data["correspondence_support"].copy()
            confidence = data["confidence"].copy()
        return field, source_support, correspondence, confidence, record

    def _edge(self, fixed, moving):
        from .transforms import save_pull_field
        current_config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if current_config_hash != self.config_hash:
            raise ValueError("Registration configuration changed during a cached graph run")
        anchors = self.fitting.get(str(moving)) if fixed == self.reference else None
        fitting_hash = hashlib.sha256(json.dumps(anchors, sort_keys=True, allow_nan=False).encode()).hexdigest()
        key = (fixed, moving, self.config_hash, fitting_hash)
        if key in self.edges:
            return self._load_edge(self.edges[key])
        reference_image, moving_image = self.images[fixed], self.images[moving]
        identical = np.array_equal(reference_image, moving_image)
        # Identical nonreference images still need the same region, exclusion,
        # landmark and feature validation as every other edge. Only the actual
        # annotation reference has a separate unconditional identity policy.
        self.attempted_fits += 1
        try:
            result = self.estimate(reference_image, moving_image, self.config,
                                   polarity=self.polarity, roi=self.roi,
                                   fitting_landmarks=anchors)
        except BaseException as exc:
            exc.registration_attempted_fits = self.attempted_fits
            exc.registration_completed_fits = self.completed_fits
            raise
        self.completed_fits += 1
        field = np.asarray(result["pull_displacement_yx"], dtype=np.float32)
        correspondence = np.asarray(result["correspondence_support"], dtype=bool)
        confidence = np.asarray(result["confidence"], dtype=np.float32)
        metrics = _json_values(dict(result["metrics"]))
        valid = bool(result["registration_valid"])
        if field.shape != (2, *self.shape) or not np.isfinite(field).all():
            raise ValueError("Registration edge returned an invalid pull displacement field")
        if (correspondence.shape != self.shape or confidence.shape != self.shape
                or not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any()):
            raise ValueError("Registration edge support/confidence must match the finite reference grid")
        source_support = _sampling_support(field, self.shape)
        correspondence &= source_support
        if not valid:
            correspondence[:] = False
        confidence = np.where(correspondence, confidence, 0).astype(np.float32)
        edge_id = f"frame-{fixed:06d}-to-{moving:06d}"
        field_file = f"transforms/edges/{edge_id}-pull.npz"
        support_file = f"transforms/edges/{edge_id}-support.npz"
        metadata = {"source_frame": moving, "reference_frame": fixed,
                    "annotation_reference_frame": self.reference,
                    "source_sha256": self.source_hash, "method": self.config["method"],
                    "reference_geometry": self.geometry, "source_geometry": self.geometry,
                    "coarse_transform_already_composed": True, "edge_id": edge_id,
                    "estimation_config_sha256": self.config_hash,
                    "fitting_landmarks_sha256": fitting_hash}
        digest = save_pull_field(self.directory/field_file, field, metadata)
        with (self.directory/support_file).open("xb") as stream:
            np.savez_compressed(stream, source_support=source_support,
                                correspondence_support=correspondence, confidence=confidence)
        record = {"edge_id": edge_id, "fixed_source_frame": fixed, "moving_source_frame": moving,
                  "field_file": field_file, "field_sha256": digest,
                  "support_file": support_file, "support_sha256": _sha256(self.directory/support_file),
                  "estimated": not identical, "identity_control": identical, "estimator_called": True,
                  "estimation_config_sha256": self.config_hash, "fitting_landmarks_sha256": fitting_hash,
                  "registration_valid": valid, "metrics": metrics}
        self.edges[key] = record
        return field, source_support, correspondence, confidence, record

    def result(self, target):
        from .transforms import (apply_pull, compose_pull_fields, field_quality,
                                 invert_pull_field, sample_support)
        route = self.routes[target]
        field = np.zeros((2, *self.shape), dtype=np.float32)
        geometric = np.ones(self.shape, dtype=bool)
        support = np.ones(self.shape, dtype=bool)
        confidence = np.ones(self.shape, dtype=np.float32)
        coarse = np.eye(3)
        records = []
        all_edges_valid = True
        for fixed, moving in zip(route[:-1], route[1:]):
            edge, edge_geometric, edge_support, edge_confidence, record = self._edge(fixed, moving)
            old_field = field
            field, geometric = compose_pull_fields(edge, old_field,
                                                   outer_valid=edge_geometric, inner_valid=geometric)
            support &= geometric & sample_support(edge_support, np.indices(self.shape, dtype=float)+old_field)
            confidence = np.minimum(confidence, _sample_through_pull(edge_confidence, old_field)).astype(np.float32)
            confidence[~support] = 0
            all_edges_valid &= record["registration_valid"]
            records.append(record)
            coarse = np.asarray(record["metrics"].get("coarse_pull_matrix_xy", np.eye(3)), dtype=float) @ coarse
        qa = field_quality(field)
        geometry_ok = (qa["min_jacobian"] >= self.config.get("min_jacobian", .2)
                       and qa["min_cell_jacobian"] >= self.config.get("min_jacobian", .2)
                       and qa["folded_cell_fraction"] == 0
                       and qa["max_local_strain"] <= self.config.get("max_local_strain", .5)
                       and qa["max_displacement_px"] <= self.config.get("max_displacement_px", 32.))
        tolerance = self.config.get("inverse_tolerance_px", .1)
        inverse, inverse_valid, inverse_residual = invert_pull_field(
            field, tolerance_px=tolerance, max_iterations=self.config.get("inverse_iterations", 80))
        roundtrip, roundtrip_valid = compose_pull_fields(inverse, field,
                                                       outer_valid=inverse_valid, inner_valid=support)
        roundtrip_error = np.linalg.norm(roundtrip, axis=0)
        support &= roundtrip_valid & (roundtrip_error <= tolerance)
        confidence[~support] = 0
        metrics = dict(records[0]["metrics"]) if len(records) == 1 else {
            "method": self.config["method"], "global_model": self.config.get("global_model", "translation"),
            "coarse_pull_matrix_xy": coarse.tolist(),
            "coarse_shift_y_px": float(-coarse[1, 2]), "coarse_shift_x_px": float(-coarse[0, 2]),
            "coarse_shift_semantics": "translation component of the composed coarse pull affine; not the final warp"}
        direct_check = None
        agreement_ok = True
        if self.direct_check and len(route) > 2:
            direct, _, direct_support, _, direct_record = self._edge(self.reference, target)
            common = support & direct_support
            check, check_valid = compose_pull_fields(inverse, direct,
                                                     outer_valid=inverse_valid, inner_valid=common)
            common &= check_valid
            error = np.linalg.norm(check, axis=0)
            count = int(common.sum())
            direct_check = {"edge_id": direct_record["edge_id"], "shared_support_pixels": count,
                            "error_space": "reference_pixels_via_checked_composed_inverse",
                            "median_disagreement_px": float(np.median(error[common])) if count else None,
                            "p95_disagreement_px": float(np.percentile(error[common], 95)) if count else None,
                            "max_disagreement_px": float(error[common].max()) if count else None,
                            "screening_limit_px": self.max_disagreement,
                            "anatomical_acceptance": False}
            agreement_ok = (direct_record["registration_valid"] and count >= 32
                            and direct_check["p95_disagreement_px"] <= self.max_disagreement)
            # A requested independent check cannot silently pass without shared evidence.
            support &= common & (error <= self.max_disagreement)
            confidence[~support] = 0
        valid = bool(all_edges_valid and geometry_ok and agreement_ok and support.sum() >= 32)
        if not valid:
            support[:] = False
            confidence[:] = 0
        metrics.update(status="review" if valid else "unsupported" if geometry_ok else "failed",
                       reason="Reference-anchored path candidate requires anatomical review" if valid else
                              "Path failed an edge, composed geometry, inverse, shared support or requested direct-consistency check",
                       qa=qa, registration_path=route,
                       registration_edge_ids=[record["edge_id"] for record in records],
                       path_length_edges=len(records),
                       inverse_qa={"converged_fraction": float(inverse_valid.mean()),
                                   "max_converged_residual_px": float(inverse_residual[inverse_valid].max()) if inverse_valid.any() else None,
                                   "supported_reference_roundtrip_p95_px": float(np.percentile(roundtrip_error[support], 95)) if support.any() else None,
                                   "independent_reverse_registration": False},
                       support_fraction=float(support.mean()), confidence_path_rule="minimum uncalibrated edge score",
                       time_assumption="none; explicit path composes coordinates and does not infer cadence or velocity")
        if direct_check is not None:
            metrics["path_direct_check"] = direct_check
        if len(records) > 1:
            ref_feature = features(self.images[self.reference], self.polarity)
            target_feature = features(self.images[target], self.polarity)
            warped_feature, sampled = apply_pull(target_feature, field)
            roi = roi_slices(self.shape, self.roi)
            metrics.update(ncc_before=correlation(ref_feature[roi], target_feature[roi]),
                           ncc_after=correlation(ref_feature[roi], warped_feature[roi], sampled[roi]),
                           ncc_scope="top-level legacy ROI, auxiliary only; no anatomical accuracy implication")
        return {"pull_displacement_yx": field, "correspondence_support": support,
                "confidence": confidence, "metrics": _json_values(metrics), "registration_valid": valid}


def _prepare_fields(source_file, output_dir, config, supplied, entry, originals,
                    header, selected, reference, scores, source_count):
    """Prepare opt-in schema-2 runs without changing the legacy translation path."""
    from .deformable import estimate_pair, validate_registration_config
    from .transforms import apply_pull, save_pull_field, field_quality

    nested = dict(config["registration"])
    fitting = _fitting_landmarks(nested.pop("fitting_landmarks", {}), selected,
                                 reference, originals.shape[1:])
    paths, routes = _registration_paths(nested.pop("paths", {}), selected, reference)
    direct_check = nested.pop("check_paths_directly", False)
    max_disagreement = nested.pop("max_path_disagreement_px", 2.)
    if type(direct_check) is not bool:
        raise ValueError("registration.check_paths_directly must be boolean")
    if (isinstance(max_disagreement, bool) or not isinstance(max_disagreement, (int, float))
            or not np.isfinite(max_disagreement) or max_disagreement <= 0):
        raise ValueError("registration.max_path_disagreement_px must be positive")
    if fitting and any(paths.values()):
        raise ValueError("Reference-frame fitting_landmarks cannot be mixed with keyframe paths without explicit edge anchors")
    normalized = validate_registration_config(nested)
    config["registration"] = {**normalized, "fitting_landmarks": fitting, "paths": paths,
                              "check_paths_directly": direct_check,
                              "max_path_disagreement_px": float(max_disagreement)}
    shape = originals.shape[1:]
    reference_pos = selected.index(reference)
    for index, points in fitting.items():
        if (np.array_equal(originals[selected.index(int(index))], originals[reference_pos])
                and not np.allclose(points["reference"], points["moving"], atol=1e-7, rtol=0)):
            raise ValueError("Nonidentity fitting landmarks conflict with pixel-identical reference images")
    source_hash = _sha256(source_file)
    geometry = {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
                for key, value in spatial_header(header).items() if key != "encoding"}
    # Hash equal frames without dropping or renumbering the original sequence.
    first_hash = {}
    duplicates = {}
    frame_hashes = {}
    for pos, index in enumerate(selected):
        digest = hashlib.sha256(np.ascontiguousarray(originals[pos]).tobytes()).hexdigest()
        frame_hashes[index] = digest
        if digest in first_hash:
            duplicates[index] = first_hash[digest]
        else:
            first_hash[digest] = index

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    aligned = np.empty(originals.shape, dtype=np.float32)
    validity = np.empty(originals.shape, dtype=bool)
    correspondence = np.empty(originals.shape, dtype=bool)
    confidence = np.empty(originals.shape, dtype=np.float32)
    frame_info = []
    graph = None
    try:
        (temporary / "transforms").mkdir()
        graph = _RegistrationGraph(originals, selected, reference, normalized,
                                   config["polarity"], config["roi"], fitting, routes,
                                   direct_check, float(max_disagreement), temporary,
                                   source_hash, geometry, estimate_pair)
        for pos, index in enumerate(selected):
            exact_reference = np.array_equal(originals[pos], originals[reference_pos])
            if index == reference:
                field = np.zeros((2, *shape), dtype=np.float32)
                support = np.ones(shape, dtype=bool)
                certainty = np.ones(shape, dtype=np.float32)
                metrics = {"status": "reference" if index == reference else "review",
                           "reason": "Fixed reference identity",
                           "ncc_before": 1.0, "ncc_after": 1.0,
                           "coarse_shift_y_px": 0.0, "coarse_shift_x_px": 0.0,
                           "qa": field_quality(field), "registration_path": [reference],
                           "registration_edge_ids": [], "path_length_edges": 0}
                mechanically_valid = True
            else:
                result = graph.result(index)
                field = np.asarray(result["pull_displacement_yx"], dtype=np.float32)
                support = np.asarray(result["correspondence_support"], dtype=bool)
                certainty = np.asarray(result["confidence"], dtype=np.float32)
                metrics = _json_values(dict(result["metrics"]))
                mechanically_valid = bool(result["registration_valid"])
            if field.shape != (2, *shape) or not np.isfinite(field).all():
                raise ValueError("Registration returned an invalid pull displacement field")
            if support.shape != shape or certainty.shape != shape or not np.isfinite(certainty).all():
                raise ValueError("Registration support/confidence must match the finite reference grid")
            if (certainty < 0).any() or (certainty > 1).any():
                raise ValueError("Registration confidence must be in 0..1")
            aligned[pos], validity[pos] = apply_pull(originals[pos], field)
            support = support & validity[pos]
            if not mechanically_valid:
                support[:] = False
            correspondence[pos] = support
            confidence[pos] = np.where(support, certainty, 0)
            relative = f"transforms/frame-{index:06d}-pull.npz"
            metadata = {"source_frame": index, "reference_frame": reference,
                        "source_sha256": source_hash, "method": normalized["method"],
                        "reference_geometry": geometry, "source_geometry": geometry,
                        "coarse_transform_already_composed": True,
                        "registration_path": metrics["registration_path"],
                        "registration_edge_ids": metrics["registration_edge_ids"]}
            field_hash = save_pull_field(temporary / relative, field, metadata)
            # A core success is only a review candidate, never an anatomical approval.
            status = "reference" if index == reference else (
                "review" if mechanically_valid else metrics.get("status", "failed"))
            if not mechanically_valid and status not in {"failed", "unsupported"}:
                status = "failed"
            dy = float(metrics.get("coarse_shift_y_px", 0.0) or 0.0)
            dx = float(metrics.get("coarse_shift_x_px", 0.0) or 0.0)
            info = {**metrics, "source_frame": index, "status": status,
                    "field_file": relative, "field_sha256": field_hash,
                    "registration_valid": mechanically_valid,
                    "annotation_eligible": index == reference,
                    "shift_y_px": dy, "shift_x_px": dx,
                    "coarse_shift_y_px": dy, "coarse_shift_x_px": dx,
                    "selection_score": scores[index], "valid_fraction": float(validity[pos].mean()),
                    "correspondence_fraction": float(support.mean()),
                    "source_frame_sha256": frame_hashes[index],
                    "identity_control": exact_reference,
                    "independent_observation": index not in duplicates and not exact_reference,
                    "included_in_minip": mechanically_valid and bool(support.any())
                                         and index not in config["exclude_from_minip"]}
            if index in duplicates:
                info["duplicate_of_source_frame"] = duplicates[index]
            frame_info.append(_json_values(info))
            del field
        include = [item["included_in_minip"] for item in frame_info]
        composite, composite_valid = valid_minip(
            aligned if config["polarity"] == "dark" else -aligned,
            validity & correspondence, include)
        if config["polarity"] == "bright":
            composite = -composite
        warnings = ["Local registration is exploratory until shared anatomy is explicitly reviewed.",
                    "Confidence and transform regularity are not anatomical accuracy or complete vessel-tree coverage.",
                    "Scalar shifts describe coarse initialization only; final transforms are saved pull fields."]
        if entry["pixel_spacing_known"] is not True:
            warnings.append("Pixel spacing is unknown; do not interpret placeholder spacing as measured anatomy.")
        if entry["frame_order_certain"] is not True:
            warnings.append("Temporal frame order is not verified; no temporal trajectory was inferred.")
        if duplicates:
            warnings.append("Pixel-identical source frames retained with original indices; they are not independent evidence.")
        dependencies = {package: version(package) for package in ("numpy", "scipy", "scikit-image", "pynrrd")}
        if normalized["method"] == "demons":
            dependencies["SimpleITK"] = version("SimpleITK")
        manifest = {"schema_version": 2, "software_version": __version__,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": entry["run_id"], "group_id": f"{entry['run_id']}:reference-{reference}",
                    "source_file": str(source_file), "source_sha256": source_hash,
                    "source_metadata_sha256": _sha256(source_file.with_suffix(".json"))
                                              if source_file.with_suffix(".json").is_file() else None,
                    "sequence_file": "registered.seq.nrrd", "reference_file": "reference.nrrd",
                    "minip_file": "minip.nrrd", "validity_file": "validity.seq.nrrd",
                    "minip_validity_file": "minip_validity.nrrd",
                    "correspondence_file": "correspondence.seq.nrrd", "confidence_file": "confidence.seq.nrrd",
                    "reference_frame": reference, "spatial_shape": list(shape),
                    "frames": frame_info, "config": config, "registration_method": normalized["method"],
                    "registration_fits": graph.attempted_fits,
                    "registration_completed_fits": graph.completed_fits,
                    "registration_fit_count_semantics": "actual estimator dispatches, including identity validation; fixed reference excluded",
                    "registration_edges": list(graph.edges.values()),
                    "registration_path_convention": "explicit fixed-to-moving pull composition, anchored to the actual reference; no timing inference",
                    "source_frame_count": source_count, "summary_frame": entry["summary_frame"],
                    "pixel_spacing_known": entry["pixel_spacing_known"], "frame_order_certain": entry["frame_order_certain"],
                    "reference_selection": "automatic" if supplied.get("reference_frame") is None else "user",
                    "review_required": True, "annotation_policy": "explicit_anatomical_review_required",
                    "projection_exploratory": True, "full_tree_coverage_established": False,
                    "warnings": warnings, "projection": "minimum" if config["polarity"] == "dark" else "maximum",
                    "shift_fields_meaning": "coarse content shift only; not the final transformation",
                    "transform_convention": "pull_displacement_yx: Phi(y,x)=(y,x)+u(y,x); components dy,dx; source pixels; coarse already composed",
                    "support_semantics": {"validity_file": "original image sampling support",
                                          "correspondence_file": "candidate shared-structure support; requires anatomical review",
                                          "confidence_file": "uncalibrated image-matching score, not probability",
                                          "reference_identity": "identity support does not establish complete vessel-tree observation"},
                    "geometry": geometry, "dependencies": dependencies,
                    "provenance": {**_code_provenance(), "seed": None,
                                   "seed_note": "Deterministic registration; no random study sampling",
                                   "evaluation_split": "exploratory; independent anatomical landmarks required"}}
        write_sequence(temporary / manifest["sequence_file"], aligned, header, selected)
        write_sequence(temporary / manifest["validity_file"], validity.astype(np.uint8), header, selected)
        write_sequence(temporary / manifest["correspondence_file"], correspondence.astype(np.uint8), header, selected)
        write_sequence(temporary / manifest["confidence_file"], confidence, header, selected)
        write_volume(temporary / manifest["reference_file"], originals[reference_pos], header)
        write_volume(temporary / manifest["minip_file"], composite, header)
        write_volume(temporary / manifest["minip_validity_file"], composite_valid.astype(np.uint8), header)
        from .projections import write_projection_outputs
        manifest["projection_comparison"] = write_projection_outputs(
            temporary, manifest, originals, aligned, validity, composite, composite_valid, header)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        (temporary / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
        with (temporary / "transforms.tsv").open("w", newline="") as stream:
            fields = ["source_frame", "coarse_shift_y_px", "coarse_shift_x_px", "status", "registration_valid",
                      "annotation_eligible", "correspondence_fraction", "field_file", "field_sha256", "reason"]
            writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(frame_info)
        from .report import write_report
        write_report(temporary, manifest, originals, aligned, validity)
        temporary.rename(output_dir)
    except BaseException as exc:
        if graph is not None:
            exc.registration_attempted_fits = graph.attempted_fits
            exc.registration_completed_fits = graph.completed_fits
        shutil.rmtree(temporary)
        raise
    return manifest

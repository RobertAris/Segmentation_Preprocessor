"""Independent landmark evaluation in the final annotation plane.

The optimizer never consumes these records. Numerical/image confidence is kept
separate from anatomical error, and absent annotations never become zero error.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

from .io import read_sequence


CONVENTION = "original_full_resolution_xy"
VISIBILITY = {"shared", "reference_only", "moving_only", "ambiguous"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Prepared artifact paths must be nonempty strings")
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Prepared artifact paths must remain inside the prepared directory")
    return path


def annotation_template(prepared_dir):
    """Return an empty, fingerprinted collection packet; no guessed landmarks."""
    root = Path(prepared_dir).expanduser().resolve()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported preparation manifest schema")
    return {"schema_version": 1, "prepared_manifest_sha256": _sha256(path),
            "coordinate_convention": CONVENTION,
            "reference_frame": manifest["reference_frame"],
            "source_frames": [item["source_frame"] for item in manifest["frames"]],
            "spatial_shape_yx": manifest["spatial_shape"],
            "instructions": "Record corresponding original [x,y] coordinates, branch identity, visibility and reference width. Keep evaluation points out of fitting. Empty annotations establish no accuracy.",
            "landmark_example": {"source_frame": None, "point_id": "branch-junction-id",
                                 "role": "evaluation", "visibility": "shared",
                                 "reference": None, "moving": None, "reference_width_px": None,
                                 "region": "proximal_or_named_distal_branch", "notes": ""},
            "landmarks": []}


def _coordinates(value, shape, name, required):
    if value is None and not required:
        return None
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite original [x,y] coordinates")
    if (result < 0).any() or result[0] > shape[1]-1 or result[1] > shape[0]-1:
        raise ValueError(f"{name} must lie inside the full-resolution original frame")
    return result


def _records(packet, manifest, manifest_hash):
    if not isinstance(packet, dict) or packet.get("schema_version") != 1:
        raise ValueError("Landmark packet must use schema_version 1")
    if packet.get("prepared_manifest_sha256") != manifest_hash:
        raise ValueError("Landmark packet preparation fingerprint does not match")
    if packet.get("coordinate_convention") != CONVENTION:
        raise ValueError("Landmarks must use original_full_resolution_xy")
    rows = packet.get("landmarks")
    if not isinstance(rows, list):
        raise ValueError("landmarks must be a list")
    available = {item["source_frame"] for item in manifest["frames"]}
    fitting = dict(manifest["config"].get("landmarks", {}))
    fitting.update(manifest["config"].get("registration", {}).get("fitting_landmarks", {}))
    seen = set()
    shape = tuple(manifest["spatial_shape"])
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every landmark must be an object")
        frame = row.get("source_frame")
        point_id = row.get("point_id")
        if type(frame) is not int or frame not in available:
            raise ValueError("Landmarks must name selected original source frames")
        if not isinstance(point_id, str) or not point_id.strip() or (frame, point_id) in seen:
            raise ValueError("point_id must be nonempty and unique within each source frame")
        seen.add((frame, point_id))
        if row.get("role") not in {"evaluation", "fitting"}:
            raise ValueError("Every point needs an explicit evaluation or fitting role")
        visible = row.get("visibility")
        if visible not in VISIBILITY:
            raise ValueError("Unknown landmark visibility")
        reference = _coordinates(row.get("reference"), shape, "reference", visible in {"shared", "reference_only"})
        moving = _coordinates(row.get("moving"), shape, "moving", visible in {"shared", "moving_only"})
        width = row.get("reference_width_px")
        if width is not None and (isinstance(width, bool) or not isinstance(width, (int, float))
                                  or not np.isfinite(width) or width <= 0):
            raise ValueError("reference_width_px must be positive when supplied")
        if row["role"] == "evaluation" and visible == "shared" and str(frame) in fitting:
            used = fitting[str(frame)]
            # Conservative coordinate-level protection in addition to explicit roles.
            if (np.any(np.linalg.norm(np.asarray(used["reference"])-reference, axis=1) < 1e-6)
                    or np.any(np.linalg.norm(np.asarray(used["moving"])-moving, axis=1) < 1e-6)):
                raise ValueError(f"Evaluation landmark {point_id} overlaps a fitting landmark")
        result.append({**row, "reference": reference, "moving": moving,
                       "reference_width_px": width})
    return result


def _sample(mask, xy):
    return float(ndimage.map_coordinates(np.asarray(mask, dtype=float),
                 np.asarray(xy, dtype=float)[::-1, None], order=1,
                 mode="constant", cval=0., prefilter=False)[0])


def _supported_at(mask, xy):
    from .transforms import sample_support
    return bool(sample_support(mask, np.asarray(xy, dtype=float)[::-1, None])[0])


def _support_sequence(root, manifest, key):
    """Read support only when its frame indices and fixed geometry are verified."""
    data, header = read_sequence(_local_file(root, manifest[key]))
    expected = [row["source_frame"] for row in manifest["frames"]]
    if data.shape != (len(expected), *manifest["spatial_shape"]):
        raise ValueError(f"{key} shape differs from the prepared reference sequence")
    if not np.isin(data, [0, 1]).all():
        raise ValueError(f"{key} must be a binary support mask")
    if manifest["schema_version"] == 2:
        if len(set(expected)) != len(expected) or any(type(index) is not int for index in expected):
            raise ValueError("Preparation source frame indices must be unique integers")
        try:
            indices = [int(value) for value in str(header["axis 3 index values"]).split()]
        except (KeyError, ValueError):
            raise ValueError(f"{key} lacks original source frame index labels") from None
        if indices != expected or len(set(indices)) != len(indices):
            raise ValueError(f"{key} source frame index labels differ from the manifest")
        if (int(header.get("dimension", 0)) != 4
                or list(header.get("kinds", [])) != ["domain", "domain", "domain", "list"]
                or list(header.get("labels", [])) != ["", "", "", "source_frame"]
                or header.get("axis 3 index type") != "numeric"):
            raise ValueError(f"{key} must have singleton-slice spatial axes and an indexed source-frame sequence")
        geometry = manifest.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError("Schema-2 preparation requires explicit reference geometry")
        directions = np.asarray(header.get("space directions"), dtype=float)
        expected_directions = np.asarray(geometry.get("space directions"), dtype=float)
        origin = np.asarray(header.get("space origin", []), dtype=float)
        expected_origin = np.asarray(geometry.get("space origin", []), dtype=float)
        if (directions.shape != (4, 3) or expected_directions.shape != (3, 3)
                or origin.shape != (3,) or expected_origin.shape != (3,)
                or not np.allclose(directions[:3], expected_directions, rtol=1e-6, atol=1e-7)
                or not np.isnan(directions[3]).all()
                or not np.allclose(origin, expected_origin, rtol=1e-6, atol=1e-7)
                or header.get("space") != geometry.get("space")):
            raise ValueError(f"{key} geometry differs from the reference annotation grid")
        if "space units" in geometry and list(header.get("space units", [])) != list(geometry["space units"]):
            raise ValueError(f"{key} physical units differ from the reference grid")
    return data.astype(bool)


def _stats(values):
    if not values:
        return {"count": 0, "median": None, "p95": None, "max": None}
    data = np.asarray(values, dtype=float)
    return {"count": len(values), "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95)), "max": float(data.max())}


def _summarize(rows):
    points = [row for row in rows if row.get("role") == "evaluation"
              and row.get("visibility") == "shared" and not row.get("identity_control")]
    measured = [row for row in points if row.get("reference_error_px") is not None]
    normalized = [row["reference_error_over_width"] for row in measured
                  if row.get("reference_error_over_width") is not None]
    errors = _stats([row["reference_error_px"] for row in measured])
    widths = _stats(normalized)
    all_eligible = bool(points) and all(row.get("eligible_for_acceptance") for row in points)
    targets_passed = (all_eligible and len(normalized) == len(points)
                      and errors["median"] <= .5 and errors["p95"] <= 1.0
                      and widths["p95"] <= .25)
    return {"shared_evaluation_landmarks": len(points), "measured_landmarks": len(measured),
            "unsupported_or_failed_landmarks": sum(not row.get("eligible_for_acceptance") for row in points),
            "landmarks_without_reference_width": sum(row.get("reference_width_px") is None for row in points),
            "reference_error_px": errors, "reference_error_over_width": widths,
            "source_pull_residual_px": _stats([row["source_pull_residual_px"] for row in points
                                               if row.get("source_pull_residual_px") is not None]),
            "quantitative_targets_passed": bool(targets_passed)}


def _frame_digest(value):
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) else None


def _identity_controls(root, manifest, needed):
    """Resolve reference copies without assuming missing metadata means independence."""
    frames = {row["source_frame"]: row for row in manifest["frames"]}
    reference = manifest["reference_frame"]
    if reference not in frames:
        raise ValueError("Preparation reference frame is absent from selected source frames")
    reference_hash = _frame_digest(frames[reference].get("source_frame_sha256"))
    controls = {reference: True}
    methods = {str(reference): "fixed_reference_identity"}
    unresolved = set()
    for index in needed:
        if index == reference:
            continue
        row = frames[index]
        digest = _frame_digest(row.get("source_frame_sha256"))
        if row.get("identity_control") is True or row.get("independent_observation") is False:
            controls[index] = True
            methods[str(index)] = "persisted_control_metadata"
        elif reference_hash is not None and digest is not None:
            controls[index] = digest == reference_hash
            methods[str(index)] = "persisted_frame_sha256"
        elif row.get("identity_control") is False and row.get("independent_observation") is True:
            controls[index] = False
            methods[str(index)] = "persisted_control_metadata"
        else:
            unresolved.add(index)
    provenance = {"methods_by_source_frame": methods,
                  "verified_original_source_file": None, "verified_original_source_sha256": None,
                  "note": "Missing identity evidence never establishes an independent observation."}
    if unresolved:
        value = manifest.get("source_file")
        if not isinstance(value, str) or not value:
            raise ValueError("Independent evaluation requires original source_file because frame identity metadata is missing")
        source = Path(value).expanduser()
        source = (root/source).resolve() if not source.is_absolute() else source.resolve()
        if not source.is_file():
            raise ValueError("Original source is unavailable; cannot establish independence without frame identity metadata")
        digest = _sha256(source)
        if digest != manifest.get("source_sha256"):
            raise ValueError("Original source SHA-256 mismatch; cannot establish frame identity")
        originals, _ = read_sequence(source)
        selected = list(frames)
        if (originals.shape[1:] != tuple(manifest["spatial_shape"])
                or any(type(index) is not int or not 0 <= index < len(originals) for index in selected)
                or (manifest.get("source_frame_count") is not None
                    and manifest["source_frame_count"] != len(originals))):
            raise ValueError("Original source grid/frame indices differ from the preparation manifest")
        # Stored source index order is irrelevant: the chosen reference can occur
        # after a pixel-identical earlier image in the selected frame list.
        for index in selected:
            if index in unresolved or index not in controls:
                controls[index] = bool(np.array_equal(originals[index], originals[reference]))
                methods[str(index)] = "verified_original_source"
        provenance.update(verified_original_source_file=str(source),
                          verified_original_source_sha256=digest)
    return controls, provenance


def evaluate_landmarks(prepared_dir, annotations):
    """Evaluate withheld correspondences without granting anatomical approval."""
    from .transforms import (field_quality, invert_pull_field, load_pull_field,
                             map_points, shift_to_pull)
    root = Path(prepared_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported preparation manifest schema")
    packet = json.loads(Path(annotations).read_text()) if isinstance(annotations, (str, Path)) else annotations
    manifest_hash = _sha256(manifest_path)
    records = _records(packet, manifest, manifest_hash)
    frames = {row["source_frame"]: row for row in manifest["frames"]}
    needed = {row["source_frame"] for row in records
              if row["role"] == "evaluation" and row["visibility"] == "shared"}
    identity_controls, identity_provenance = _identity_controls(root, manifest, needed)
    source_validity = _support_sequence(root, manifest, "validity_file")
    if manifest["schema_version"] == 2:
        correspondence = _support_sequence(root, manifest, "correspondence_file")
    else:
        correspondence = source_validity
    cache = {}
    result = []
    positions = {row["source_frame"]: pos for pos, row in enumerate(manifest["frames"])}
    config = manifest.get("config", {}).get("registration", {})
    tolerance = float(config.get("inverse_tolerance_px", .1))
    iterations = int(config.get("inverse_iterations", 80))
    for row in records:
        frame = frames[row["source_frame"]]
        output = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
        output.update(source_pull_residual_px=None, reference_error_px=None,
                      reference_error_over_width=None, inverse_roundtrip_residual_px=None,
                      eligible_for_acceptance=False,
                      identity_control=identity_controls.get(row["source_frame"],
                          row["source_frame"] == manifest["reference_frame"]
                          or frame.get("identity_control") is True
                          or frame.get("independent_observation") is False))
        if row["role"] != "evaluation" or row["visibility"] != "shared":
            output["measurement_status"] = "excluded_fitting" if row["role"] == "fitting" else "excluded_visibility"
            result.append(output)
            continue
        index = row["source_frame"]
        if index not in cache:
            if manifest["schema_version"] == 2:
                field_path = _local_file(root, frame["field_file"])
                if _sha256(field_path) != frame["field_sha256"]:
                    raise ValueError(f"Pull-field hash mismatch for source frame {index}")
                field, metadata = load_pull_field(field_path)
                if field.shape[1:] != tuple(manifest["spatial_shape"]):
                    raise ValueError("Pull-field grid differs from the reference annotation grid")
                if (metadata.get("source_frame") != index
                        or metadata.get("reference_frame") != manifest["reference_frame"]
                        or metadata.get("source_sha256") != manifest["source_sha256"]):
                    raise ValueError("Pull-field source/reference provenance mismatch")
            else:
                field = shift_to_pull(tuple(manifest["spatial_shape"]),
                                      [frame["shift_y_px"], frame["shift_x_px"]])
            qa = field_quality(field)
            numerical_valid = qa["min_jacobian"] > 0 and qa["min_cell_jacobian"] > 0
            inverse, inverse_support, residual = invert_pull_field(
                field, tolerance_px=tolerance, max_iterations=iterations)
            cache[index] = (field, inverse, inverse_support, residual, numerical_valid)
        field, inverse, inverse_support, residual, numerical_valid = cache[index]
        mapped, domain = map_points(field, row["reference"][None])
        source_residual = float(np.linalg.norm(mapped[0] - row["moving"]))
        output["source_pull_residual_px"] = source_residual
        inverse_point, inverse_domain = map_points(inverse, row["moving"][None])
        roundtrip, inverse_target_domain = map_points(field, inverse_point)
        roundtrip_error = float(np.linalg.norm(roundtrip[0] - row["moving"]))
        inverse_checked = (numerical_valid and bool(inverse_domain[0]) and bool(inverse_target_domain[0])
                           and _supported_at(inverse_support, row["moving"])
                           and _sample(residual, row["moving"]) <= tolerance
                           and roundtrip_error <= tolerance + 1e-6)
        output["inverse_roundtrip_residual_px"] = roundtrip_error
        if inverse_checked:
            error = float(np.linalg.norm(inverse_point[0] - row["reference"]))
            output["reference_error_px"] = error
            if row["reference_width_px"] is not None:
                output["reference_error_over_width"] = error / row["reference_width_px"]
        pos = positions[index]
        supported = (bool(domain[0]) and _supported_at(source_validity[pos], row["reference"])
                     and _supported_at(correspondence[pos], row["reference"])
                     and _supported_at(source_validity[pos], inverse_point[0])
                     and _supported_at(correspondence[pos], inverse_point[0]))
        valid_frame = frame.get("registration_valid", frame.get("status") not in {"rejected", "uninformative", "failed", "unsupported"})
        output["eligible_for_acceptance"] = bool(inverse_checked and supported and valid_frame)
        output["measurement_status"] = ("measured" if output["eligible_for_acceptance"] else
                                         "unsupported_correspondence" if inverse_checked else "invalid_inverse")
        result.append(output)
    per_frame = {str(index): _summarize([row for row in result if row["source_frame"] == index])
                 for index in sorted({row["source_frame"] for row in result})}
    regions = sorted({str(row.get("region", "unspecified")) for row in result})
    per_region = {region: _summarize([row for row in result if str(row.get("region", "unspecified")) == region])
                  for region in regions}
    summary = _summarize(result)
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_manifest_sha256": manifest_hash,
            "evaluator_source_sha256": _sha256(Path(__file__)),
            "identity_control_verification": identity_provenance,
            "landmark_packet_sha256": _sha256(annotations) if isinstance(annotations, (str, Path)) else
                                      hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "status": "requires_annotations" if not summary["shared_evaluation_landmarks"] else "evaluated",
            "coordinate_convention": CONVENTION,
            "primary_error_space": "reference_annotation_pixels_via_checked_inverse",
            "inverse_tolerance_px": tolerance,
            "targets": {"median_reference_error_px": .5, "p95_reference_error_px": 1.,
                        "p95_reference_error_over_width": .25,
                        "status": "proposed_engineering_targets_not_validated_anatomical_thresholds"},
            "summary": summary, "per_frame": per_frame, "per_region": per_region,
            "measurement_status_counts": dict(Counter(row["measurement_status"] for row in result)),
            "landmarks": result, "anatomical_acceptance_established": False,
            "full_tree_coverage_established": False,
            "limitations": ["Landmark sampling does not establish accuracy between points or full-tree coverage.",
                            "Quantitative targets require reader-repeatability assessment and anatomical review.",
                            "Identity and duplicate-reference controls are excluded from the primary accuracy summary."]}

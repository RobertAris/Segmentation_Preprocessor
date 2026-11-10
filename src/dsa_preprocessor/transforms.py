"""Pixel-domain pull maps with explicit support and numerical quality checks.

Fields have shape (2, reference_y, reference_x), components [dy, dx], and
Phi(q) = q + field(q). They sample the *original* source image. A legacy
moving-content shift s therefore corresponds to the constant pull field -s.
These checks establish numerical behavior, never anatomical correspondence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def _field(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 2 or min(value.shape[1:]) < 2:
        raise ValueError("Pull displacement must have shape (2,height,width), each axis >=2")
    if not np.isfinite(value).all():
        raise ValueError("Pull displacement must be finite")
    return value


def identity_grid(shape):
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Expected a 2D grid with each axis >=2")
    return np.indices(shape, dtype=np.float64)


def shift_to_pull(shape, shift_yx):
    shift = np.asarray(shift_yx, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("Content shift must be finite [dy,dx]")
    return np.broadcast_to(-shift[:, None, None], (2, *shape)).astype(np.float32).copy()


def _inside(coords, shape):
    return ((coords[0] >= 0) & (coords[0] <= shape[0]-1) &
            (coords[1] >= 0) & (coords[1] <= shape[1]-1))


def _sample_field(field, coords):
    return np.stack([ndimage.map_coordinates(component, coords, order=1,
                     mode="nearest", prefilter=False) for component in field])


def sample_support(mask, coordinates_yx):
    """Require all nonzero bilinear contributors, including subpixel edge cases.

    Sampling a float mask and comparing it with nearly one can accidentally allow
    a tiny contribution from forbidden pixels. Checking floor/ceil vertices avoids
    that threshold ambiguity and also rejects coordinates outside the mask grid.
    """
    mask=np.asarray(mask,dtype=bool)
    coords=np.asarray(coordinates_yx,dtype=np.float64)
    if mask.ndim!=2 or min(mask.shape)<1 or coords.ndim<1 or coords.shape[0]!=2:
        raise ValueError("Support requires a 2D mask and [y,x] coordinate arrays")
    finite=np.isfinite(coords).all(axis=0)
    inside=finite&_inside(coords,mask.shape)
    y=np.clip(np.where(finite,coords[0],0),0,mask.shape[0]-1)
    x=np.clip(np.where(finite,coords[1],0),0,mask.shape[1]-1)
    y0,y1=np.floor(y).astype(int),np.ceil(y).astype(int)
    x0,x1=np.floor(x).astype(int),np.ceil(x).astype(int)
    return inside&mask[y0,x0]&mask[y0,x1]&mask[y1,x0]&mask[y1,x1]


def apply_pull(image, field):
    """Sample original intensities once; invalid borders have a neutral fill."""
    field = _field(field)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D source image")
    coords = identity_grid(field.shape[1:]) + field
    valid = _inside(coords, image.shape)
    if image.shape == field.shape[1:] and not np.any(field):
        return image.copy(), valid
    result = ndimage.map_coordinates(image, coords, order=1, mode="constant",
                                    cval=float(np.median(image)), prefilter=False)
    return result.astype(np.float32), valid


def compose_pull_fields(outer, inner, *, outer_valid=None, inner_valid=None):
    """Return outer o inner, including interpolation-domain support.

    If inner maps reference -> intermediate and outer maps intermediate -> source,
    their composition samples the original source. Fields are never simply added.
    Optional support masks require every interpolation contributor to be valid.
    """
    outer, inner = _field(outer), _field(inner)
    coords = identity_grid(inner.shape[1:]) + inner
    valid = _inside(coords, outer.shape[1:])
    result = inner + _sample_field(outer, coords)
    if outer_valid is not None:
        mask = np.asarray(outer_valid, dtype=bool)
        if mask.shape != outer.shape[1:]:
            raise ValueError("outer_valid shape does not match outer field")
        valid &= sample_support(mask,coords)
    if inner_valid is not None:
        mask = np.asarray(inner_valid, dtype=bool)
        if mask.shape != inner.shape[1:]:
            raise ValueError("inner_valid shape does not match inner field")
        valid &= mask
    return result.astype(np.float32), valid


def affine_to_pull(shape, matrix_xy):
    """Convert a 3x3 homogeneous reference->source [x,y] affine to a pull field."""
    matrix = np.asarray(matrix_xy, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError("Expected a finite 2D affine homogeneous matrix")
    grid = identity_grid(shape)
    coords = np.stack([grid[1], grid[0], np.ones(shape)])
    mapped = np.einsum("ij,jhw->ihw", matrix, coords)
    return (mapped[[1, 0]] - grid).astype(np.float32)


def map_points(field, points_xy):
    """Map Nx2 [x,y] points; return mapped points and field-domain support."""
    field = _field(field)
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be finite Nx2 [x,y]")
    coords = points[:, ::-1].T
    supported = _inside(coords, field.shape[1:])
    displacement = _sample_field(field, coords).T[:, ::-1]
    return points + displacement, supported


def _derivatives(field):
    uy_y, uy_x = np.gradient(field[0].astype(float))
    ux_y, ux_x = np.gradient(field[1].astype(float))
    return 1+uy_y, uy_x, ux_y, 1+ux_x


def cell_jacobians(field):
    """Four oriented triangle areas per cell, covering both grid diagonals.

    Identity has four values of one. A nonpositive value detects pixel-scale
    folding that a central-difference determinant may miss.
    """
    field = _field(field)
    p = identity_grid(field.shape[1:]) + field
    p00, p10 = p[:, :-1, :-1], p[:, 1:, :-1]
    p01, p11 = p[:, :-1, 1:], p[:, 1:, 1:]
    def det(a, b):
        return a[0]*b[1]-a[1]*b[0]
    return np.stack([det(p10-p00, p01-p00), det(p11-p10, p00-p10),
                     det(p01-p11, p10-p11), det(p00-p01, p11-p01)])


def field_quality(field):
    field = _field(field)
    a,b,c,d = _derivatives(field)
    determinant = a*d-b*c
    triangles = cell_jacobians(field)
    trace = a*a+b*b+c*c+d*d
    discriminant = np.maximum(trace*trace-4*determinant*determinant, 0)
    max_sv = np.sqrt(np.maximum((trace+np.sqrt(discriminant))/2, 0))
    min_sv = np.sqrt(np.maximum((trace-np.sqrt(discriminant))/2, 0))
    strain = np.maximum(np.abs(max_sv-1), np.abs(min_sv-1))
    return {"min_jacobian": float(determinant.min()),
            "min_cell_jacobian": float(triangles.min()),
            "folded_cell_fraction": float((triangles.min(axis=0) <= 0).mean()),
            "max_local_strain": float(strain.max()),
            "max_displacement_px": float(np.linalg.norm(field, axis=0).max())}


def invert_pull_field(field, *, tolerance_px=0.1, max_iterations=80, source_shape=None):
    """Numerically invert a map; never confuse convergence with invertibility.

    Returns (inverse displacement on source grid, converged support, residual).
    Only pixels with an in-domain solution, positive local orientation, and a
    residual <= tolerance are marked valid. The estimator separately rejects
    folded maps. This inverse is not an independently estimated reverse fit.
    """
    field = _field(field)
    tolerance = tolerance_px
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Positive inverse tolerance and iteration count required")
    shape = field.shape[1:] if source_shape is None else tuple(source_shape)
    target = identity_grid(shape)
    point = target - _sample_field(field, target)
    derivatives = np.stack(_derivatives(field))
    for _ in range(int(max_iterations)):
        residual_vec = point + _sample_field(field, point) - target
        a,b,c,d = [ndimage.map_coordinates(v, point, order=1, mode="nearest",
                    prefilter=False) for v in derivatives]
        det = a*d-b*c
        good = np.abs(det) > 1e-8
        safe = np.where(good, det, 1.)
        step = np.stack([(d*residual_vec[0]-b*residual_vec[1])/safe,
                         (-c*residual_vec[0]+a*residual_vec[1])/safe])
        step[:, ~good] = 0
        length = np.maximum(np.linalg.norm(step, axis=0)/8., 1.)
        point -= step/length
        if float(np.max(np.linalg.norm(step,axis=0))) < tolerance*.1:
            break
    residual = np.linalg.norm(point+_sample_field(field,point)-target,axis=0)
    inside = _inside(point, field.shape[1:])
    triangles = cell_jacobians(field).min(axis=0)
    safe_cells = triangles > 0
    safe_pixels = np.ones(field.shape[1:], dtype=bool)
    for dy,dx in [(0,0),(1,0),(0,1),(1,1)]:
        safe_pixels[dy:dy+safe_cells.shape[0], dx:dx+safe_cells.shape[1]] &= safe_cells
    local_safe = sample_support(safe_pixels,point)
    valid = inside & local_safe & np.isfinite(residual) & (residual <= tolerance)
    return (point-target).astype(np.float32), valid, residual.astype(np.float32)


_CONTRACT = {"field_format_version": 1, "representation": "displacement",
             "direction": "reference_to_source", "components": ["dy", "dx"],
             "units": "source_pixels", "mapping": "Phi(q)=q+u(q)"}


def save_pull_field(path, field, metadata):
    """Save a new NPZ without pickle, protected against accidental replacement."""
    field = _field(field)
    meta = dict(metadata)
    for key,value in _CONTRACT.items():
        if key in meta and meta[key] != value:
            raise ValueError(f"Conflicting field metadata: {key}")
    meta.update(_CONTRACT, shape_yx=list(field.shape[1:]))
    encoded = json.dumps(meta, sort_keys=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, pull_displacement_yx=field,
                            metadata_json=np.asarray(encoded))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pull_field(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"pull_displacement_yx", "metadata_json"}:
            raise ValueError("Unsupported pull-field file contents")
        raw = data["pull_displacement_yx"]
        if raw.dtype != np.float32:
            raise ValueError("Persisted pull field must be float32")
        field = _field(raw).copy()
        meta = json.loads(str(data["metadata_json"].item()))
    if any(meta.get(k) != v for k,v in _CONTRACT.items()):
        raise ValueError("Unsupported pull-field coordinate contract")
    if meta.get("shape_yx") != list(field.shape[1:]):
        raise ValueError("Field metadata shape mismatch")
    return field, meta

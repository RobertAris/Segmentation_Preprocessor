"""Conservative translation estimates; correlation scores are review aids only."""
import warnings

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def roi_slices(shape, roi=None):
    h, w = shape
    if roi is None:
        x, y = int(w * .1), int(h * .1)
        rw, rh = w - 2*x, h - 2*y
    else:
        if len(roi) != 4 or any(int(v) != v for v in roi):
            raise ValueError("ROI must be integer [x, y, width, height]")
        x, y, rw, rh = map(int, roi)
    if min(rw, rh) < 8 or x < 0 or y < 0 or x+rw > w or y+rh > h:
        raise ValueError("ROI must be at least 8x8 and inside the source frame")
    return slice(y, y+rh), slice(x, x+rw)


def features(image, polarity="dark"):
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [.5, 99.5])
    if hi-lo < 1e-8:
        return np.zeros_like(image)
    image = np.clip((image-lo)/(hi-lo), 0, 1)
    fine = ndimage.gaussian_filter(image, .8)
    background = ndimage.gaussian_filter(image, 6)
    response = background-fine if polarity == "dark" else fine-background
    # Local contrast suppresses broad illumination changes without modifying annotation intensities.
    return np.maximum(response, 0)


def correlation(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    else:
        a, b = a.ravel(), b.ravel()
    if a.size < 32:
        return None
    a, b = a.astype(np.float64), b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denom) if denom > 1e-12 else None


def apply_shift(image, shift):
    """Apply moving->reference (dy, dx) once, linear interpolation, no wrap."""
    dy, dx = map(float, shift)
    if not np.isfinite([dy, dx]).all():
        raise ValueError("Shift must be finite")
    image = np.asarray(image, dtype=np.float32)
    if dy == 0 and dx == 0:
        return image.copy(), np.ones(image.shape, dtype=bool)
    yy = np.arange(image.shape[0], dtype=float)[:, None]-dy
    xx = np.arange(image.shape[1], dtype=float)[None, :]-dx
    valid = (yy >= 0) & (yy <= image.shape[0]-1) & (xx >= 0) & (xx <= image.shape[1]-1)
    fill = float(np.median(image))
    result = ndimage.shift(image, (dy, dx), order=1, mode="constant", cval=fill, prefilter=False)
    return result, valid


def valid_minip(aligned, validity, include=None):
    aligned, validity = np.asarray(aligned), np.asarray(validity, dtype=bool)
    if aligned.shape != validity.shape or aligned.ndim != 3:
        raise ValueError("Aligned data and validity must have matching (time,y,x) shapes")
    if include is not None:
        include = np.asarray(include, dtype=bool)
        if include.shape != (len(aligned),):
            raise ValueError("include must have one value per frame")
        validity = validity & include[:, None, None]
    valid = validity.any(axis=0)
    result = np.min(np.where(validity, aligned, np.inf), axis=0)
    # Never display an uncovered border as an artificial dark vessel.
    samples = aligned[validity]
    fill = float(np.percentile(samples, 99.5)) if samples.size else float(np.median(aligned))
    result[~valid] = fill
    return result.astype(np.float32), valid


def register_frames(frames, config):
    frames = np.asarray(frames)
    if frames.ndim != 3 or len(frames) < 1 or not np.isfinite(frames).all():
        raise ValueError("Expected nonempty finite (time,y,x) frames")
    reference_index = int(config.get("reference_index", 0))
    if not 0 <= reference_index < len(frames):
        raise ValueError("reference_index is out of range")
    roi = roi_slices(frames.shape[1:], config.get("roi"))
    polarity = config.get("polarity", "dark")
    if polarity not in ("dark", "bright"):
        raise ValueError("polarity must be dark or bright")
    max_shift = float(config.get("max_shift_px", 32))
    upsample = int(config.get("upsample_factor", 10))
    threshold = float(config.get("min_correlation", .35))
    normalization = config.get("normalization", "none")
    if max_shift <= 0 or not np.isfinite(max_shift) or not 1 <= upsample <= 100:
        raise ValueError("max_shift_px must be positive; upsample_factor must be 1..100")
    if normalization not in (None, "none", "phase") or not -1 <= threshold <= 1:
        raise ValueError("Invalid normalization or min_correlation")
    ref = features(frames[reference_index], polarity)[roi]
    informative_ref = float(ref.std()) > 1e-6
    shifts = np.zeros((len(frames), 2), dtype=float)
    metrics = []
    for index, image in enumerate(frames):
        info = {"status": "reference" if index == reference_index else "estimated",
                "ncc_before": None, "ncc_after": None, "estimated_shift_y_px": 0.,
                "estimated_shift_x_px": 0., "reason": ""}
        if index == reference_index:
            if not informative_ref:
                info.update(status="uninformative", reason="Reference ROI has no usable local contrast")
            else:
                info.update(ncc_before=1., ncc_after=1.)
            metrics.append(info)
            continue
        moving = features(image, polarity)[roi]
        if not informative_ref or moving.std() < 1e-6:
            info.update(status="uninformative", reason="Insufficient local contrast in matching ROI")
            metrics.append(info)
            continue
        before = correlation(ref, moving)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            shift, _, _ = phase_cross_correlation(ref, moving, upsample_factor=upsample,
                                                  normalization="phase" if normalization == "phase" else None)
        info.update(ncc_before=before, estimated_shift_y_px=float(shift[0]), estimated_shift_x_px=float(shift[1]))
        if not np.isfinite(shift).all() or np.linalg.norm(shift) > max_shift:
            info.update(status="rejected", reason="Estimated displacement exceeds max_shift_px; identity retained")
            metrics.append(info)
            continue
        aligned_features, valid = apply_shift(moving, shift)
        after = correlation(ref, aligned_features, valid)
        info["ncc_after"] = after
        shifts[index] = shift
        if after is None or after < threshold or (before is not None and after < before-.02):
            info.update(status="review", reason="Low or worsened ROI correlation; excluded from composite")
        metrics.append(info)
    return shifts, metrics

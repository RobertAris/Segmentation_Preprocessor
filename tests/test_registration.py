"""Numerical tests for the image-to-reference translation convention."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift

from dsa_preprocessor.registration import apply_shift, register_frames, valid_minip


def vessel_phantom(shape=(128, 160)):
    """Asymmetric branched dark structures away from all image edges."""
    signal = np.zeros(shape, dtype=np.float32)
    for x0, y0, x1, y1 in [(30, 96, 78, 70), (78, 70, 115, 25),
                            (78, 70, 126, 80), (58, 81, 37, 38),
                            (101, 42, 130, 35)]:
        count = int(2 * np.hypot(x1-x0, y1-y0))
        xx = np.linspace(x0, x1, count).round().astype(int)
        yy = np.linspace(y0, y1, count).round().astype(int)
        signal[yy, xx] = 1
    return (1000 - 450 * gaussian_filter(signal, 1.2)).astype(np.float32)


def settings(**overrides):
    return dict(reference_index=0, max_shift_px=20, upsample_factor=20,
                normalization='none', polarity='dark', min_correlation=0.25,
                **overrides)


def test_recovers_subpixel_image_to_reference_shift():
    reference = vessel_phantom()
    displacement = np.array([3.25, -4.5])
    moving = nd_shift(reference, displacement, order=1, mode='nearest', prefilter=False)
    shifts, metrics = register_frames(np.stack([reference, moving]), settings())
    np.testing.assert_array_equal(shifts[0], [0, 0])
    np.testing.assert_allclose(shifts[1], -displacement, atol=0.4)
    aligned, valid = apply_shift(moving, shifts[1])
    assert aligned.dtype == np.float32
    assert valid.dtype == bool
    # Only assess the original field of view and omit the source padding.
    roi = np.s_[12:-12, 12:-12]
    before = np.mean((moving[roi] - reference[roi]) ** 2)
    after = np.mean((aligned[roi] - reference[roi]) ** 2)
    assert after < before * 0.15


def test_zero_shift_preserves_original_pixels():
    original = np.arange(11*17, dtype=np.uint16).reshape(11, 17)
    aligned, valid = apply_shift(original, (0, 0))
    np.testing.assert_array_equal(aligned, original)
    assert valid.all()


def test_translation_does_not_wrap_and_has_explicit_validity():
    image = np.full((10, 14), 1000, dtype=np.uint16)
    image[3, 4] = 5
    aligned, valid = apply_shift(image, (2, -3))
    assert aligned[5, 1] == 5
    assert not valid[:2].any()
    assert not valid[:, -3:].any()
    assert valid[2:, :-3].all()
    assert not (aligned[valid] == 0).any()


def test_minip_never_selects_invalid_dark_padding():
    frames = np.array([[[0, 20, 0], [40, 50, 0]],
                       [[10, 0, 0], [30, 60, 0]]], dtype=np.float32)
    validity = np.array([[[False, True, False], [True, True, False]],
                         [[True, False, False], [True, True, False]]])
    composite, valid = valid_minip(frames, validity)
    np.testing.assert_array_equal(composite[:, :2], [[10, 20], [30, 50]])
    np.testing.assert_array_equal(valid, [[True, True, False], [True, True, False]])
    assert np.isfinite(composite).all()
    # Uncovered pixels also need a neutral display fill, never an artificial dark vessel.
    assert (composite[:, -1] >= 20).all()


def test_excluded_frame_cannot_contribute_to_composite():
    frames = np.array([[[10, 20]], [[1, 2]]], dtype=np.float32)
    validity = np.ones_like(frames, dtype=bool)
    composite, valid = valid_minip(frames, validity, include=[True, False])
    np.testing.assert_array_equal(composite, [[10, 20]])
    assert valid.all()


def test_constant_moving_frame_is_never_accepted():
    reference = vessel_phantom()
    frames = np.stack([reference, np.full_like(reference, 1000)])
    try:
        shifts, metrics = register_frames(frames, settings())
    except ValueError:
        return
    assert metrics[1]['status'] not in {'accepted', 'reference', 'manual', 'ok'}


def test_shift_beyond_limit_is_never_silently_accepted():
    reference = vessel_phantom()
    moving = nd_shift(reference, (0, 12), order=1, mode='nearest', prefilter=False)
    config = settings()
    config['max_shift_px'] = 5
    try:
        shifts, metrics = register_frames(np.stack([reference, moving]), config)
    except ValueError:
        return
    assert metrics[1]['status'] not in {'accepted', 'reference', 'manual', 'ok'}


def test_fractional_border_is_invalid_when_any_sample_would_be_outside():
    aligned, valid = apply_shift(np.full((9, 13), 1000, dtype=np.uint16), (0.5, -0.5))
    assert not valid[0].any()
    assert not valid[:, -1].any()
    assert valid[1:, :-1].all()
    np.testing.assert_allclose(aligned[valid], 1000)

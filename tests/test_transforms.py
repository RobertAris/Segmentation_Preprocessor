"""Independent coordinate checks for final pull fields and unsupported inverses."""
import json

import numpy as np
import pytest

from dsa_preprocessor.registration import apply_shift
from dsa_preprocessor.transforms import (affine_to_pull, apply_pull, cell_jacobians,
    compose_pull_fields, field_quality, identity_grid, invert_pull_field,
    load_pull_field, map_points, sample_support, save_pull_field, shift_to_pull)


def test_legacy_shift_matches_pull_and_fractional_support():
    image=np.arange(19*31,dtype=np.uint16).reshape(19,31)
    field=shift_to_pull(image.shape,(2.5,-3.25))
    actual,valid=apply_pull(image,field)
    expected,expected_valid=apply_shift(image,(2.5,-3.25))
    np.testing.assert_array_equal(valid,expected_valid)
    np.testing.assert_allclose(actual,expected,atol=1e-5)
    assert not valid[:3].any() and not valid[:,-4:].any()


def test_identity_is_exact_and_bad_arrays_rejected():
    image=np.arange(9*13,dtype=np.uint16).reshape(9,13)
    actual,valid=apply_pull(image,np.zeros((2,9,13),np.float32))
    np.testing.assert_array_equal(actual,image)
    assert valid.all()
    with pytest.raises(ValueError,match="finite"):
        apply_pull(image,np.full((2,9,13),np.nan))


def test_composition_order_is_affine_after_local_and_support_is_conservative():
    shape=(23,35)
    matrix=np.array([[1.04,.02,1.],[-.01,.98,-.5],[0,0,1.]])
    outer=affine_to_pull(shape,matrix)
    inner=shift_to_pull(shape,(-.4,.6))
    composed,supported=compose_pull_fields(outer,inner)
    grid=identity_grid(shape)
    intermediate=grid+inner
    expected=np.stack([matrix[1,0]*intermediate[1]+matrix[1,1]*intermediate[0]+matrix[1,2],
                       matrix[0,0]*intermediate[1]+matrix[0,1]*intermediate[0]+matrix[0,2]])-grid
    np.testing.assert_allclose(composed[:,supported],expected[:,supported],atol=1e-5)
    assert not supported[-1].any()
    assert not supported[:,0].any()
    assert np.max(np.abs(composed-(outer+inner)))>.005
    invalid=np.ones(shape,bool);invalid[10,10]=False
    _,masked=compose_pull_fields(outer,np.zeros_like(inner),outer_valid=invalid)
    assert not masked[10,10]


def test_inverse_matches_analytic_affine_on_non_square_grid():
    shape=(41,67)
    matrix=np.array([[1.03,.04,-1.],[.01,.98,1.5],[0,0,1.]])
    field=affine_to_pull(shape,matrix)
    inverse,valid,residual=invert_pull_field(field,tolerance_px=.001)
    expected=affine_to_pull(shape,np.linalg.inv(matrix))
    assert valid.mean()>.85
    np.testing.assert_allclose(inverse[:,valid],expected[:,valid],atol=.001)
    assert residual[valid].max()<=.001
    points=np.array([[22.2,16.7],[40.5,24.6]])
    mapped,inside=map_points(field,points)
    recovered,reverse_inside=map_points(inverse,mapped)
    assert inside.all() and reverse_inside.all()
    np.testing.assert_allclose(recovered,points,atol=.002)


def test_inverse_marks_outside_domain_unsupported():
    field=shift_to_pull((17,29),(-2.,3.))
    inverse,valid,residual=invert_pull_field(field)
    assert not valid[:2].any()
    assert not valid[:,-3:].any()
    assert valid[2:,:-3].all()
    np.testing.assert_allclose(inverse[:,valid],-field[:,valid],atol=1e-6)


def test_four_triangle_check_detects_pixel_scale_fold():
    field=np.zeros((2,9,13),np.float32)
    assert np.all(cell_jacobians(field)==1)
    field[1,4,6]=3.
    # The central difference at the displaced vertex ignores its own displacement.
    ux_y,ux_x=np.gradient(field[1])
    assert 1+ux_x[4,6]==1
    qa=field_quality(field)
    assert qa['folded_cell_fraction']>0
    assert qa['min_cell_jacobian']<0
    _,valid,_=invert_pull_field(field)
    assert not valid.all()


def test_serialization_contract_and_overwrite_protection(tmp_path):
    field=shift_to_pull((7,11),(1.25,-.5))
    path=tmp_path/'field.npz'
    digest=save_pull_field(path,field,{'source_frame':17,'reference_frame':2})
    restored,metadata=load_pull_field(path)
    np.testing.assert_array_equal(restored,field)
    assert len(digest)==64
    assert metadata['direction']=='reference_to_source'
    assert metadata['source_frame']==17
    with pytest.raises(FileExistsError):save_pull_field(path,field,{})
    with pytest.raises(ValueError,match='Conflicting'):
        save_pull_field(tmp_path/'bad.npz',field,{'units':'millimetres'})
    metadata['components']=['dx','dy']
    np.savez(tmp_path/'corrupt.npz',pull_displacement_yx=field,metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError,match='coordinate contract'):
        load_pull_field(tmp_path/'corrupt.npz')


def test_support_rejects_even_tiny_forbidden_interpolation_contribution():
    mask=np.ones((13,19),bool);mask[:,10:]=False
    coordinates=np.array([[5.,5.,5.,5.],[9.,9.+1e-9,-1e-9,10.]])
    np.testing.assert_array_equal(sample_support(mask,coordinates),[True,False,False,False])

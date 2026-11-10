"""Synthetic known-motion checks; these do not validate real anatomy."""
import importlib.util

import numpy as np
import pytest
from scipy import ndimage

from dsa_preprocessor.deformable import estimate_pair, validate_registration_config
from dsa_preprocessor.correspondence import (local_correspondences,patch_supported,
    region_mask,source_support_mask,_guide_displacement)
from dsa_preprocessor.transforms import (affine_to_pull, apply_pull, identity_grid,
                                         invert_pull_field, map_points,sample_support)


def phantom(shape=(176,208)):
    yy,xx=np.indices(shape)
    rng=np.random.default_rng(29)
    signal=np.zeros(shape)
    points=[]
    for y in [35,65,100,135]:
        for x in [38,72,110,145,173]:
            y0=y+rng.uniform(-4,4);x0=x+rng.uniform(-4,4)
            signal+=rng.uniform(.5,1.)*np.exp(-((yy-y0)**2/(rng.uniform(2,5)**2)+(xx-x0)**2/(rng.uniform(2,5)**2))/2)
            points.append([x0,y0])
    return (1200-450*signal).astype(np.float32),np.asarray(points)


def settings(**overrides):
    return {"method":"bspline","patch_size":15,"patch_step":12,"search_radius_px":7,
            "min_match_ncc":.5,"min_peak_margin":.005,"min_matches":4,
            "grid_spacings_px":[64,32],"support_radius_px":30.,
            "max_displacement_px":16.,"max_local_strain":.7,**overrides}


@pytest.mark.parametrize('method',['translation','rigid','similarity','affine','bspline','tvl1'])
def test_recovers_known_subpixel_translation_without_changing_inputs(method):
    reference,points=phantom()
    original=reference.copy()
    displacement=np.array([2.25,-3.5])
    moving=ndimage.shift(reference,displacement,order=1,mode='nearest',prefilter=False)
    moving_original=moving.copy()
    result=estimate_pair(reference,moving,settings(method=method))
    assert result['registration_valid'],result['metrics']
    predicted,inside=map_points(result['pull_displacement_yx'],points)
    error=np.linalg.norm(predicted-(points+displacement[::-1]),axis=1)
    assert np.median(error)<.6,(method,error,result['metrics'])
    assert result['metrics']['status']=='review'
    assert result['metrics']['anatomical_acceptance'] is False
    assert result['correspondence_support'].any()
    np.testing.assert_array_equal(reference,original)
    np.testing.assert_array_equal(moving,moving_original)


def test_bspline_resolves_spatially_varying_known_motion():
    reference,points=phantom()
    grid=identity_grid(reference.shape)
    field=np.zeros((2,*reference.shape),np.float32)
    field[0]=3*np.sin(2*np.pi*grid[1]/reference.shape[1])*np.sin(np.pi*grid[0]/reference.shape[0])
    field[1]=2*np.cos(np.pi*grid[1]/reference.shape[1])*np.sin(np.pi*grid[0]/reference.shape[0])
    inverse,valid,_=invert_pull_field(field,tolerance_px=.001)
    moving,_=apply_pull(reference,inverse)
    global_result=estimate_pair(reference,moving,settings(method='translation'))
    local_result=estimate_pair(reference,moving,settings(regularization=.1))
    assert local_result['registration_valid'],local_result['metrics']
    true_points,_=map_points(field,points)
    global_points,_=map_points(global_result['pull_displacement_yx'],points)
    local_points,_=map_points(local_result['pull_displacement_yx'],points)
    error_global=np.median(np.linalg.norm(global_points-true_points,axis=1))
    error_local=np.median(np.linalg.norm(local_points-true_points,axis=1))
    assert error_local<error_global*.5,(error_global,error_local)
    assert error_local<.65


@pytest.mark.parametrize('method',['rigid','similarity','affine'])
def test_global_models_recover_true_rotation_and_scale(method):
    reference,points=phantom()
    angle=np.deg2rad(2.)
    scale=1.025 if method!='rigid' else 1.
    linear=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center=np.array(reference.shape[::-1])/2
    matrix=np.eye(3);matrix[:2,:2]=linear
    matrix[:2,2]=center-linear@center+np.array([1.5,-1.])
    inverse=affine_to_pull(reference.shape,np.linalg.inv(matrix))
    moving,_=apply_pull(reference,inverse)
    result=estimate_pair(reference,moving,settings(method=method))
    assert result['registration_valid'],result['metrics']
    expected=points@matrix[:2,:2].T+matrix[:2,2]
    actual,_=map_points(result['pull_displacement_yx'],points)
    assert np.median(np.linalg.norm(actual-expected,axis=1))<.55


def test_stationary_artifact_exclusion_does_not_drive_vessel_motion():
    vessel,points=phantom()
    moving=ndimage.shift(vessel,(2.,-3.),order=1,mode='nearest')
    # An extremely contrasted stationary device occupies the excluded strip.
    reference=vessel.copy();reference[:,8:13]=100
    moving[:,8:13]=100
    result=estimate_pair(reference,moving,settings(method='bspline',
        regions=[[22,18,170,140]],exclude_regions=[[0,0,20,176]]))
    assert result['registration_valid'],result['metrics']
    actual,_=map_points(result['pull_displacement_yx'],points)
    assert np.median(np.linalg.norm(actual-(points+[-3,2]),axis=1))<.6
    assert not result['correspondence_support'][:,:20].any()


def test_missing_signal_abstains_and_duplicate_control_is_explicit():
    reference,_=phantom()
    absent=np.full_like(reference,1200)
    result=estimate_pair(reference,absent,settings())
    assert not result['registration_valid']
    assert not result['correspondence_support'].any()
    assert result['metrics']['status']=='unsupported'
    duplicate=estimate_pair(reference,reference.copy(),settings())
    assert duplicate['metrics']['duplicate_identity'] is True
    np.testing.assert_array_equal(duplicate['pull_displacement_yx'],0)


def test_disjoint_region_support_never_fills_whole_field():
    reference,points=phantom()
    moving=ndimage.shift(reference,(2,-3),order=1,mode='nearest')
    result=estimate_pair(reference,moving,settings(regions=[[18,18,85,140],[112,18,80,140]],
                                                    exclude_regions=[[18,18,30,30]],support_radius_px=15))
    assert result['registration_valid'],result['metrics']
    assert not result['correspondence_support'][:18].any()
    assert not result['correspondence_support'][:,103:112].any()
    assert not result['correspondence_support'][18:48,18:48].any()


@pytest.mark.parametrize('bad',[{'max_displacement_px':float('nan')}, {'patch_size':14},
    {'grid_spacings_px':[32,64]}, {'method':'perfect'}, {'typo':1}, {'inverse_iterations':True}])
def test_bad_configuration_is_rejected(bad):
    with pytest.raises(ValueError):validate_registration_config(bad)


def test_fitting_landmarks_have_distinct_strict_semantics():
    reference,points=phantom()
    with pytest.raises(ValueError,match='Fitting landmarks'):
        estimate_pair(reference,reference,settings(),fitting_landmarks={'reference':[[1,2]],'moving':[[1,2]]})


def test_optional_demons_dependency_is_explicit():
    if importlib.util.find_spec('SimpleITK') is not None:
        pytest.skip('Dependency is installed')
    reference,_=phantom()
    with pytest.raises(ValueError,match='SimpleITK'):
        estimate_pair(reference,reference.copy(),settings(method='demons'))


def test_demons_known_translation_when_available():
    if importlib.util.find_spec('SimpleITK') is None:
        pytest.skip('Optional dependency is absent')
    reference,points=phantom()
    moving=ndimage.shift(reference,(2,-3),order=1,mode='nearest')
    result=estimate_pair(reference,moving,settings(method='demons',demons_iterations=20))
    assert result['registration_valid'],result['metrics']
    actual,_=map_points(result['pull_displacement_yx'],points)
    assert np.median(np.linalg.norm(actual-(points+[-3,2]),axis=1))<.6


def test_straight_repeated_pattern_has_no_distinctive_two_dimensional_correspondence():
    yy,xx=np.indices((128,160))
    reference=(1200-300*np.cos(xx/4)).astype(np.float32)
    moving=ndimage.shift(reference,(3,2),order=1,mode='nearest')
    result=estimate_pair(reference,moving,settings())
    assert not result['registration_valid']
    assert not result['correspondence_support'].any()


@pytest.mark.parametrize('scales',[[1],[2,1]])
def test_matches_cannot_enter_excluded_source_content(scales):
    rng=np.random.default_rng(127)
    reference=rng.uniform(size=(90,120)).astype(np.float32)
    moving=ndimage.shift(reference,(0,8),order=0,mode='nearest')
    config=validate_registration_config(settings(patch_size=9,patch_step=4,
        search_radius_px=12,min_match_ncc=.9,min_peak_margin=.02,pyramid_scales=scales))
    mask,labels=region_mask(reference.shape,[[0,0,120,90]],[[50,0,70,90]])
    allowed=source_support_mask(reference.shape,[[50,0,70,90]])
    p,q,w,detail=local_correspondences(reference,moving,mask,labels,config,allowed)
    assert len(p)>5,detail
    assert (q[:,0]<50).all()
    assert all(patch_supported(allowed,point[::-1],4) for point in q)
    assert (np.abs(q[:,0]-p[:,0]-8)<.2).all()
    # Before the fix, valid reference centers42/45 matched forbidden source50/53.
    assert (p[:,0]<38).all()


def test_refined_subpixel_template_support_is_stricter_than_rounding():
    allowed=np.ones((40,60),bool);allowed[:,50:]=False
    assert patch_supported(allowed,np.array([20.,45.]),4)
    assert not patch_supported(allowed,np.array([20.,45.+1e-8]),4)


def test_transformed_moving_exclusions_require_explicit_source_support():
    reference=np.random.default_rng(83).uniform(size=(90,120)).astype(np.float32)
    config=validate_registration_config(settings(exclude_regions=[[50,0,70,90]]))
    mask,labels=region_mask(reference.shape,[],config['exclude_regions'])
    with pytest.raises(ValueError,match='source support'):
        local_correspondences(reference,reference,mask,labels,config)


def test_final_pull_support_excludes_original_source_exclusions():
    reference,_=phantom()
    moving=ndimage.shift(reference,(0,8),order=1,mode='nearest')
    excluded=[[110,0,98,176]]
    result=estimate_pair(reference,moving,settings(method='translation',
        exclude_regions=excluded,search_radius_px=12))
    assert result['registration_valid'],result['metrics']
    source_allowed=source_support_mask(reference.shape,excluded)
    final_positions=identity_grid(reference.shape)+result['pull_displacement_yx']
    mapped_allowed=sample_support(source_allowed,final_positions)
    reference_mask,_=region_mask(reference.shape,[],excluded)
    assert (reference_mask&~mapped_allowed).any()
    assert not (result['correspondence_support']&~mapped_allowed).any()


def test_fitting_landmark_cannot_sample_excluded_source_content():
    reference,_=phantom()
    moving=ndimage.shift(reference,(0,8),order=1,mode='nearest')
    with pytest.raises(ValueError,match='excluded'):
        estimate_pair(reference,moving,settings(exclude_regions=[[110,0,98,176]]),
            fitting_landmarks={'reference':[[30,30],[50,80],[90,120]],
                               'moving':[[38,30],[58,80],[110,120]]})


def test_pyramid_recovers_large_distinct_local_motions_with_fine_evidence():
    rng=np.random.default_rng(821)
    reference=np.zeros((192,240),np.float32)
    reference[20:172,20:100]=rng.uniform(size=(152,80))
    reference[20:172,140:220]=rng.uniform(size=(152,80))
    moving=np.zeros_like(reference)
    moving[20:172,30:110]=reference[20:172,20:100]
    moving[20:172,130:210]=reference[20:172,140:220]
    mask,labels=region_mask(reference.shape,[[20,20,80,152],[140,20,80,152]],[])
    config=validate_registration_config(settings(search_radius_px=6,patch_size=15,
        patch_step=12,min_match_ncc=.94,min_peak_margin=.03,support_radius_px=48.))
    p0,q0,_,_=local_correspondences(reference,moving,mask,labels,config)
    p,q,_,detail=local_correspondences(reference,moving,mask,labels,
                                      {**config,'pyramid_scales':[2,1]})
    assert len(p)>20,detail
    assert len(p)>len(p0)+20,(len(p0),len(p))
    left=p[:,0]<120
    assert left.any() and (~left).any()
    expected=np.where(left,10.,-10.)
    assert np.max(np.abs((q-p)[:,0]-expected))<.2
    assert np.max(np.abs((q-p)[:,1]))<.2
    assert detail['finest_evidence_scale']==1
    assert [level['scale'] for level in detail['pyramid_levels']]==[2,1]


@pytest.mark.parametrize('scales',[[2],[1,2],[2,2,1],[3,1],[True,1],[16,8,4,2,1]])
def test_invalid_pyramid_configuration_rejected(scales):
    with pytest.raises(ValueError,match='pyramid_scales'):
        validate_registration_config({'pyramid_scales':scales})


def test_coarse_prior_requires_enclosing_consistent_support():
    reference=np.array([[10.,10.],[10.,30.],[30.,10.],[30.,30.]])
    moving=reference+[10.,0.]
    np.testing.assert_allclose(_guide_displacement([20,20],reference,moving,50,2,15),[10,0])
    assert _guide_displacement([40,20],reference,moving,50,2,15) is None
    assert _guide_displacement([20,20],reference,moving,50,2,5) is None
    moving[-1]+=[30.,0.]
    assert _guide_displacement([20,20],reference,moving,50,2,50) is None

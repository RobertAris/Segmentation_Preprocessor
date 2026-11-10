"""Integration checks independent of the local optimizer's matching behavior."""
import hashlib
import json
import sys
import types

import nrrd
import numpy as np
import pytest
from scipy.ndimage import shift

from dsa_preprocessor import pipeline
from dsa_preprocessor.io import read_sequence
from dsa_preprocessor.transforms import load_pull_field, shift_to_pull


@pytest.fixture
def source(tmp_path):
    yy, xx = np.mgrid[:48, :64]
    pattern = (1200-500*np.exp(-((xx-25)**2/3+(yy-22)**2/60))).astype(np.uint16)
    data = np.stack([pattern, shift(pattern, (2,-3), order=0, mode='nearest'), pattern, pattern])
    path = tmp_path/'sample_angio.nrrd'
    header = {'space':'left-posterior-superior',
              'space directions':np.array([[0.,.2,0.],[-.3,0.,0.],[0.,0.,1.]]),
              'space origin':[2.,4.,8.], 'kinds':['domain']*3}
    nrrd.write(str(path), data, header, index_order='C')
    path.with_suffix('.json').write_text(json.dumps({'PixelSpacingKnown':True,'FrameOrderCertain':True}))
    return path, data, header


def fake_core(monkeypatch, estimate):
    module = types.ModuleType('dsa_preprocessor.deformable')
    module.validate_registration_config = lambda value: {'method':'bspline',**value}
    module.estimate_pair = estimate
    monkeypatch.setitem(sys.modules, 'dsa_preprocessor.deformable', module)


def known_result(ref, moving, config, **kwargs):
    shape = moving.shape
    support = np.ones(shape, bool)
    support[:, :5] = False
    return {'pull_displacement_yx':shift_to_pull(shape,[-2,3]),
            'correspondence_support':support, 'confidence':support.astype(np.float32)*.8,
            'metrics':{'status':'review','reason':'Known fixture transform','ncc_before':.3,'ncc_after':.9,
                       'coarse_shift_y_px':-2.,'coarse_shift_x_px':3.},
            'registration_valid':True}


def test_schema2_retains_geometry_indices_and_single_original_resampling(source, tmp_path, monkeypatch):
    from dsa_preprocessor import transforms
    path, data, header = source
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    fake_core(monkeypatch,known_result)
    real_apply = transforms.apply_pull
    calls = []
    def counted(image, field):
        calls.append(np.asarray(image).copy())
        return real_apply(image, field)
    monkeypatch.setattr(transforms,'apply_pull',counted)
    output = tmp_path/'prepared'
    manifest = pipeline.prepare(path,output,{'frames':[1,3],'reference_frame':3,'registration':{'method':'bspline'}})
    assert manifest['schema_version']==2
    assert [row['source_frame'] for row in manifest['frames']]==[1,3]
    assert len(calls)==2
    np.testing.assert_array_equal(calls[0],data[1])
    np.testing.assert_array_equal(calls[1],data[3])
    aligned,h = read_sequence(output/manifest['sequence_file'])
    np.testing.assert_array_equal(aligned[0,8:-8,8:-8],aligned[1,8:-8,8:-8])
    assert h['axis 3 index values']=='1 3'
    np.testing.assert_allclose(h['space directions'][:2],header['space directions'][:2])
    np.testing.assert_allclose(h['space origin'],header['space origin'])
    row=manifest['frames'][0]
    field,metadata=load_pull_field(output/row['field_file'])
    np.testing.assert_allclose(field[:,10,10],[2,-3])
    assert metadata['source_frame']==1 and metadata['reference_frame']==3
    assert metadata['coarse_transform_already_composed'] is True
    assert row['field_sha256']==hashlib.sha256((output/row['field_file']).read_bytes()).hexdigest()
    assert row['annotation_eligible'] is False
    assert manifest['frames'][1]['annotation_eligible'] is True
    support,_=read_sequence(output/manifest['correspondence_file'])
    confidence,_=read_sequence(output/manifest['confidence_file'])
    assert not support[0,:,:5].any()
    assert (confidence[0,:,:5]==0).all()
    assert manifest['full_tree_coverage_established'] is False
    assert manifest['projection_exploratory'] is True
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash


def test_reference_duplicates_are_identity_controls_without_renumbering(source,tmp_path,monkeypatch):
    from dsa_preprocessor.deformable import estimate_pair
    path,_,_=source
    calls=[]
    def checked(*args,**kwargs):
        calls.append(1)
        return estimate_pair(*args,**kwargs)
    fake_core(monkeypatch,checked)
    output=tmp_path/'duplicates'
    manifest=pipeline.prepare(path,output,{'frames':[0,2,3],'reference_frame':0,'registration':{}})
    assert manifest['registration_fits']==len(calls)==2
    assert [r['source_frame'] for r in manifest['frames']]==[0,2,3]
    assert manifest['frames'][1]['duplicate_of_source_frame']==0
    assert manifest['frames'][1]['independent_observation'] is False
    assert manifest['frames'][1]['annotation_eligible'] is False
    field,_=load_pull_field(output/manifest['frames'][1]['field_file'])
    assert not field.any()


def test_unsupported_field_cannot_contribute_to_projection(source,tmp_path,monkeypatch):
    path,_,_=source
    def unsupported(*args,**kwargs):
        result=known_result(*args,**kwargs)
        result['registration_valid']=False
        result['metrics']['status']='unsupported'
        return result
    fake_core(monkeypatch,unsupported)
    manifest=pipeline.prepare(path,tmp_path/'unsupported',{'frames':[0,1],'reference_frame':0,'registration':{}})
    assert manifest['frames'][1]['registration_valid'] is False
    assert manifest['frames'][1]['correspondence_fraction']==0
    assert manifest['frames'][1]['included_in_minip'] is False


def test_estimator_failure_leaves_no_partial_output(source,tmp_path,monkeypatch):
    path,_,_=source
    def broken(*args,**kwargs):raise RuntimeError('fixture estimation failure')
    fake_core(monkeypatch,broken)
    output=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='fixture'):
        pipeline.prepare(path,output,{'frames':[0,1],'reference_frame':0,'registration':{}})
    assert not output.exists()
    assert not list(tmp_path.glob('.failed-*'))


@pytest.mark.parametrize('extra',[
    {'manual_shifts':{'1':[0,0]}},
    {'landmarks':{'1':{'reference':[[2,2]],'moving':[[2,2]]}}},
    {'upsample_factor':11},
    {'max_shift_px':8},
])
def test_new_mode_rejects_ambiguous_legacy_settings(source,tmp_path,extra):
    path,_,_=source
    with pytest.raises(ValueError,match='mix|legacy'):
        pipeline.prepare(path,tmp_path/'conflict',{'frames':[0,1],'reference_frame':0,'registration':{},**extra})
    assert not (tmp_path/'conflict').exists()


def test_schema1_remains_default(source,tmp_path):
    path,_,_=source
    manifest=pipeline.prepare(path,tmp_path/'legacy',{'frames':[0,1],'reference_frame':0})
    assert manifest['schema_version']==1
    assert 'registration' not in manifest['config']
    assert 'field_file' not in manifest['frames'][1]


def test_cli_registration_optin_is_explicit(monkeypatch,tmp_path):
    from dsa_preprocessor.cli import main
    captured={}
    def prepare(path,output,config):
        captured.update(config)
        return {'frames':[{'status':'reference'}]}
    monkeypatch.setattr(pipeline,'prepare',prepare)
    assert main(['prepare','input.nrrd','-o',str(tmp_path/'out'),'--registration-method','tvl1'])==0
    assert captured['registration']=={'method':'tvl1'}


def test_pilot_counts_reverse_and_retry_jobs_before_launch(source,tmp_path,monkeypatch):
    from dsa_preprocessor import pilot
    path,_,_=source
    packet={'schema_version':1,'max_fits':1,'jobs':[
        {'job_id':'forward','source_file':str(path),'purpose':'development','config':{'frames':[0,1],'reference_frame':0}},
        {'job_id':'reverse','source_file':str(path),'purpose':'reverse','config':{'frames':[0,1],'reference_frame':1}}]}
    packet_path=tmp_path/'packet.json';packet_path.write_text(json.dumps(packet))
    monkeypatch.setattr(pilot,'prepare',lambda *args,**kwargs:pytest.fail('Budget rejection must occur before execution'))
    with pytest.raises(ValueError,match='exceeding'):
        pilot.run_pilot(packet_path,tmp_path/'pilot')
    assert not (tmp_path/'pilot').exists()


def test_pilot_records_failures_and_preserves_budget(source,tmp_path,monkeypatch):
    from dsa_preprocessor import pilot
    path,_,_=source
    packet={'schema_version':1,'max_fits':2,'jobs':[
        {'job_id':'first','source_file':str(path),'purpose':'smoke','config':{'frames':[0,1],'reference_frame':0}},
        {'job_id':'retry','source_file':str(path),'purpose':'retry','config':{'frames':[0,1],'reference_frame':0}}]}
    packet_path=tmp_path/'packet.json';packet_path.write_text(json.dumps(packet))
    count=[]
    def fake_prepare(*args):
        count.append(1)
        if len(count)==1:
            exc=RuntimeError('recorded fixture failure');exc.registration_attempted_fits=1;exc.registration_completed_fits=0
            raise exc
        return {'source_sha256':'fixture','registration_fits':1,'registration_completed_fits':1,
                'frames':[{'status':'reference'},{'status':'estimated'}]}
    monkeypatch.setattr(pilot,'prepare',fake_prepare)
    result=pilot.run_pilot(packet_path,tmp_path/'pilot')
    assert result['status']=='completed_with_failures'
    assert result['attempted_pair_fits']==2
    assert result['completed_pair_fits']==1
    assert len(result['jobs'])==2 and result['jobs'][0]['status']=='failed'
    with pytest.raises(ValueError,match='exists'):
        pilot.run_pilot(packet_path,tmp_path/'pilot')


@pytest.mark.parametrize('through_keyframe',[False,True])
def test_actual_estimator_roundtrips_through_saved_fields_and_evaluation(tmp_path,through_keyframe):
    from scipy.ndimage import gaussian_filter
    from dsa_preprocessor.evaluation import annotation_template,evaluate_landmarks
    signal=np.zeros((128,160),np.float32)
    for x0,y0,x1,y1 in [(30,96,78,70),(78,70,115,25),(78,70,126,80),(58,81,37,38),(101,42,130,35)]:
        n=int(2*np.hypot(x1-x0,y1-y0))
        xx=np.linspace(x0,x1,n).round().astype(int);yy=np.linspace(y0,y1,n).round().astype(int)
        signal[yy,xx]=1
    reference=1000-450*gaussian_filter(signal,1.2)
    target=2 if through_keyframe else 1
    frames=np.stack([shift(reference,(2*i,-3*i),order=1,mode='nearest',prefilter=False) for i in range(target+1)])
    path=tmp_path/'known_angio.nrrd'
    header={'space':'left-posterior-superior','space directions':np.diag([.2,.3,1.]),
            'space origin':[0.,0.,0.],'kinds':['domain']*3}
    nrrd.write(str(path),frames,header,index_order='C')
    output=tmp_path/'known-prepared'
    registration={'method':'translation','patch_size':21,'patch_step':12,'min_matches':3,'support_radius_px':50}
    if through_keyframe:registration.update(paths={'2':[1]},check_paths_directly=True)
    manifest=pipeline.prepare(path,output,{'frames':list(range(target+1)),'reference_frame':0,'registration':registration})
    assert manifest['frames'][target]['registration_valid'] is True
    assert manifest['registration_fits']==(3 if through_keyframe else 1)
    annotations=annotation_template(output)
    annotations['landmarks']=[{'source_frame':target,'point_id':'junction','role':'evaluation','visibility':'shared',
        'reference':[78.,70.],'moving':[78.-3*target,70.+2*target],'reference_width_px':5.,'region':'central'}]
    evaluation=evaluate_landmarks(output,annotations)
    assert evaluation['landmarks'][0]['reference_error_px'] < .2
    assert evaluation['summary']['unsupported_or_failed_landmarks']==0
    assert evaluation['anatomical_acceptance_established'] is False


def test_earlier_copy_of_later_reference_is_still_an_identity_control(source,tmp_path,monkeypatch):
    from dsa_preprocessor.deformable import estimate_pair
    path,_,_=source
    fake_core(monkeypatch,estimate_pair)
    manifest=pipeline.prepare(path,tmp_path/'later-reference',{'frames':[0,2],'reference_frame':2,'registration':{}})
    assert manifest['frames'][0]['identity_control'] is True
    assert manifest['frames'][0]['independent_observation'] is False


def test_nonidentity_fitting_points_on_duplicate_are_rejected_atomically(source,tmp_path,monkeypatch):
    path,_,_=source
    fake_core(monkeypatch,lambda *args,**kwargs:pytest.fail('Conflicting identity constraints must fail first'))
    output=tmp_path/'bad-identity-fit'
    with pytest.raises(ValueError,match='Nonidentity fitting landmarks'):
        pipeline.prepare(path,output,{'frames':[0,2],'reference_frame':0,
            'registration':{'fitting_landmarks':{'2':{'reference':[[20,20],[25,20],[20,25]],
                                                    'moving':[[25,20],[30,20],[25,25]]}}}})
    assert not output.exists()


@pytest.fixture
def graph_source(tmp_path):
    yy,xx=np.mgrid[:48,:64]
    reference=(1200-500*np.exp(-((xx-25)**2/3+(yy-22)**2/60))).astype(np.uint16)
    data=np.stack([reference+i for i in range(4)])
    path=tmp_path/'graph_angio.nrrd'
    header={'space':'left-posterior-superior','space directions':np.diag([.2,.3,1.]),
            'space origin':[0.,0.,0.],'kinds':['domain']*3}
    nrrd.write(str(path),data,header,index_order='C')
    return path,data


def graph_estimator(data,calls,*,bad_edge=None,inconsistent_direct=False,scales=False):
    from dsa_preprocessor.transforms import affine_to_pull
    a=np.array([[1.,0,2.5],[0,1.,1.],[0,0,1.]])
    b=np.array([[1.1,0,1.],[0,1.,-1.],[0,0,1.]])
    if scales:a=b=np.diag([1.4,1.,1.])
    matrices={(0,1):a,(1,2):b,(0,2):b@a,(2,3):np.eye(3),(1,3):np.eye(3)}
    if inconsistent_direct:matrices[(0,2)]=np.array([[1.,0,14.],[0,1.,0.],[0,0,1.]])
    def estimate(reference,moving,config,**kwargs):
        key=(int(reference[0,0])-int(data[0,0,0]),int(moving[0,0])-int(data[0,0,0]))
        calls.append(key)
        matrix=matrices[key]
        support=np.ones(reference.shape,bool)
        confidence=np.full(reference.shape,.8,np.float32)
        if key==(1,2):
            support[:,40:]=False
            confidence[:]=np.linspace(0,1,reference.shape[1])[None]
        return {'pull_displacement_yx':affine_to_pull(reference.shape,matrix),
                'correspondence_support':support,'confidence':confidence,
                'metrics':{'status':'review','reason':'Known fixture edge','ncc_before':.5,'ncc_after':.8,
                           'coarse_shift_y_px':float(-matrix[1,2]),'coarse_shift_x_px':float(-matrix[0,2]),
                           'coarse_pull_matrix_xy':matrix.tolist()},
                'registration_valid':key!=bad_edge}
    return estimate,a,b


def test_keyframe_path_composes_affine_maps_and_propagates_support_once(graph_source,tmp_path,monkeypatch):
    from dsa_preprocessor import transforms
    from dsa_preprocessor.transforms import affine_to_pull
    path,data=graph_source;calls=[]
    estimate,a,b=graph_estimator(data,calls)
    fake_core(monkeypatch,estimate)
    real_apply=transforms.apply_pull;raw_resamplings=[]
    def counted(image,field):
        for index,original in enumerate(data):
            if np.array_equal(image,original):raw_resamplings.append(index)
        return real_apply(image,field)
    monkeypatch.setattr(transforms,'apply_pull',counted)
    output=tmp_path/'graph-prepared'
    manifest=pipeline.prepare(path,output,{'frames':[0,1,2],'reference_frame':0,
                                         'registration':{'paths':{'2':[1]}}})
    assert calls==[(0,1),(1,2)]
    assert sorted(raw_resamplings)==[0,1,2]
    assert manifest['registration_fits']==manifest['registration_completed_fits']==2
    assert len(manifest['registration_edges'])==2
    row=manifest['frames'][2]
    assert row['registration_path']==[0,1,2]
    assert row['registration_valid'] is True
    field,metadata=load_pull_field(output/row['field_file'])
    expected=affine_to_pull(data.shape[1:],b@a)
    np.testing.assert_allclose(field[:,10:35,5:35],expected[:,10:35,5:35],atol=1e-5)
    assert field[1,20,30]==pytest.approx(6.75,abs=1e-5)
    assert metadata['registration_path']==[0,1,2]
    support,_=read_sequence(output/manifest['correspondence_file'])
    confidence,_=read_sequence(output/manifest['confidence_file'])
    assert support[2,20,36] and not support[2,20,37]
    assert confidence[2,20,20]==pytest.approx(22.5/63,abs=1e-6)
    for edge in manifest['registration_edges']:
        assert edge['field_sha256']==hashlib.sha256((output/edge['field_file']).read_bytes()).hexdigest()
        assert edge['support_sha256']==hashlib.sha256((output/edge['support_file']).read_bytes()).hexdigest()


def test_failed_edge_cannot_be_hidden_by_later_success(graph_source,tmp_path,monkeypatch):
    path,data=graph_source;calls=[]
    estimate,_,_=graph_estimator(data,calls,bad_edge=(0,1));fake_core(monkeypatch,estimate)
    output=tmp_path/'unsupported-path'
    manifest=pipeline.prepare(path,output,{'frames':[0,1,2],'reference_frame':0,
                                         'registration':{'paths':{'2':[1]}}})
    assert calls==[(0,1),(1,2)]
    assert manifest['frames'][2]['registration_valid'] is False
    assert manifest['frames'][2]['annotation_eligible'] is False
    assert manifest['frames'][2]['correspondence_fraction']==0
    assert manifest['registration_fits']==2


def test_composed_strain_is_rechecked_even_when_individual_edges_pass(graph_source,tmp_path,monkeypatch):
    path,data=graph_source;calls=[]
    estimate,_,_=graph_estimator(data,calls,scales=True);fake_core(monkeypatch,estimate)
    manifest=pipeline.prepare(path,tmp_path/'strained-path',{'frames':[0,1,2],'reference_frame':0,
        'registration':{'paths':{'2':[1]},'max_displacement_px':128}})
    assert manifest['frames'][1]['registration_valid'] is True
    assert manifest['frames'][2]['registration_valid'] is False
    assert manifest['frames'][2]['qa']['max_local_strain'] > .5


@pytest.mark.parametrize('inconsistent',[False,True])
def test_optional_direct_path_check_is_a_real_additional_fit(graph_source,tmp_path,monkeypatch,inconsistent):
    path,data=graph_source;calls=[]
    estimate,_,_=graph_estimator(data,calls,inconsistent_direct=inconsistent);fake_core(monkeypatch,estimate)
    manifest=pipeline.prepare(path,tmp_path/'checked-path',{'frames':[0,1,2],'reference_frame':0,
        'registration':{'paths':{'2':[1]},'check_paths_directly':True}})
    assert calls==[(0,1),(1,2),(0,2)]
    assert manifest['registration_fits']==3
    row=manifest['frames'][2]
    assert row['path_direct_check']['shared_support_pixels'] > 32
    assert row['path_direct_check']['anatomical_acceptance'] is False
    assert row['registration_valid'] is (not inconsistent)


@pytest.mark.parametrize('paths',[{'2':[0]},{'2':[2]},{'2':[1,1]},{'2':[99]},{'0':[1]}])
def test_invalid_keyframe_paths_fail_before_output(graph_source,tmp_path,monkeypatch,paths):
    path,data=graph_source;fake_core(monkeypatch,lambda *a,**k:pytest.fail('Invalid paths must fail before estimation'))
    with pytest.raises(ValueError,match='path|Path'):
        pipeline.prepare(path,tmp_path/'bad-path',{'frames':[0,1,2],'reference_frame':0,'registration':{'paths':paths}})
    assert not (tmp_path/'bad-path').exists()


def test_global_fitting_points_cannot_be_reinterpreted_as_edge_anchors(graph_source,tmp_path,monkeypatch):
    path,data=graph_source;fake_core(monkeypatch,lambda *a,**k:pytest.fail('Conflicting path anchors must fail before estimation'))
    with pytest.raises(ValueError,match='fitting_landmarks cannot be mixed'):
        pipeline.prepare(path,tmp_path/'bad-anchor',{'frames':[0,1,2],'reference_frame':0,
            'registration':{'paths':{'2':[1]},'fitting_landmarks':{'2':{
                'reference':[[20,20],[25,20],[20,25]],'moving':[[22,20],[27,20],[22,25]]}}}})


def test_pilot_reservation_counts_unique_path_edges_and_direct_checks(graph_source,tmp_path):
    from dsa_preprocessor.pilot import _validate_packet
    path,_=graph_source
    job={'job_id':'paths','source_file':str(path),'purpose':'development',
         'config':{'frames':[0,1,2,3],'reference_frame':0,'registration':{'paths':{'2':[1],'3':[2]}}}}
    jobs,_,total=_validate_packet({'schema_version':1,'max_fits':4,'jobs':[job]},tmp_path,None)
    assert total==4
    assert jobs[0]['reserved_directed_edges']==[[0,1],[0,2],[1,2],[2,3]]
    job['config']['registration']['check_paths_directly']=True
    with pytest.raises(ValueError,match='exceeding'):
        _validate_packet({'schema_version':1,'max_fits':4,'jobs':[job]},tmp_path,None)


@pytest.fixture
def identical_texture_source(tmp_path):
    image=np.random.default_rng(42).uniform(0,1000,(64,80)).astype(np.float32)
    path=tmp_path/'identity-texture_angio.nrrd'
    header={'space':'left-posterior-superior','space directions':np.eye(3),
            'space origin':[0.,0.,0.],'kinds':['domain']*3}
    nrrd.write(str(path),np.stack([image,image]),header,index_order='C')
    return path,image,header


def test_identical_nonreference_uses_the_same_region_and_exclusion_support(identical_texture_source,tmp_path):
    path,_,_=identical_texture_source
    output=tmp_path/'masked-identity'
    manifest=pipeline.prepare(path,output,{'frames':[0,1],'reference_frame':0,
        'registration':{'regions':[[0,0,60,64]],'exclude_regions':[[0,0,20,64]]}})
    support,_=read_sequence(output/manifest['correspondence_file'])
    confidence,_=read_sequence(output/manifest['confidence_file'])
    assert support[0].all()  # Fixed reference retains its explicit identity policy.
    assert not support[1,:,:20].any() and not support[1,:,60:].any()
    assert support[1,:,20:60].all()
    assert not confidence[1,:,:20].any() and not confidence[1,:,60:].any()
    assert manifest['registration_fits']==manifest['registration_completed_fits']==1
    assert manifest['registration_edges'][0]['estimated'] is False
    assert manifest['registration_edges'][0]['estimator_called'] is True


def test_identical_nonreference_also_respects_top_level_roi(identical_texture_source,tmp_path):
    path,_,_=identical_texture_source
    output=tmp_path/'roi-identity'
    manifest=pipeline.prepare(path,output,{'frames':[0,1],'reference_frame':0,'roi':[20,8,40,40],
                                         'registration':{'method':'translation'}})
    support,_=read_sequence(output/manifest['correspondence_file'])
    expected=np.zeros((64,80),bool);expected[8:48,20:60]=True
    np.testing.assert_array_equal(support[1],expected)


def test_identity_consistent_landmarks_inside_exclusion_are_rejected(identical_texture_source,tmp_path):
    path,_,_=identical_texture_source
    output=tmp_path/'excluded-identity-anchor'
    anchors=[[5,5],[10,15],[15,5]]
    with pytest.raises(ValueError,match='excluded source/reference content') as error:
        pipeline.prepare(path,output,{'frames':[0,1],'reference_frame':0,
            'registration':{'regions':[[0,0,60,64]],'exclude_regions':[[0,0,20,64]],
                            'fitting_landmarks':{'1':{'reference':anchors,'moving':anchors}}}})
    assert error.value.registration_attempted_fits==1
    assert error.value.registration_completed_fits==0
    assert not output.exists() and not list(tmp_path.glob('.excluded-identity-anchor-*'))


def test_pipeline_and_core_agree_on_minimum_fitting_anchor_count(identical_texture_source,tmp_path):
    path,_,_=identical_texture_source
    with pytest.raises(ValueError,match='N>=3'):
        pipeline.prepare(path,tmp_path/'two-anchors',{'frames':[0,1],'reference_frame':0,
            'registration':{'fitting_landmarks':{'1':{'reference':[[25,25],[30,30]],'moving':[[25,25],[30,30]]}}}})
    assert not (tmp_path/'two-anchors').exists()


def test_composed_identity_edge_does_not_reintroduce_unmatched_regions(identical_texture_source,tmp_path,monkeypatch):
    from dsa_preprocessor.deformable import estimate_pair
    from dsa_preprocessor.correspondence import region_mask,source_support_mask
    from dsa_preprocessor.transforms import affine_to_pull,sample_support
    path,image,header=identical_texture_source
    nrrd.write(str(path),np.stack([image,image+1,image+1]),header,index_order='C')
    def checked(reference,moving,config,**kwargs):
        if np.array_equal(reference,moving):
            return estimate_pair(reference,moving,config,**kwargs)
        # Controlled first edge maps permitted reference columns20..59 to
        # source40..79. Source positions outside the reference matching ROI are
        # permitted unless explicitly excluded, then constrained by the next edge.
        matrix=np.array([[1.,0,20.],[0,1.,0.],[0,0,1.]])
        field=affine_to_pull(reference.shape,matrix)
        support,_=region_mask(reference.shape,config['regions'],config['exclude_regions'],kwargs.get('roi'))
        allowed=source_support_mask(moving.shape,config['exclude_regions'])
        support &= sample_support(allowed,np.indices(reference.shape,dtype=float)+field)
        return {'pull_displacement_yx':field,'correspondence_support':support,
                'confidence':support.astype(np.float32),'registration_valid':True,
                'metrics':{'status':'review','ncc_before':.5,'ncc_after':.8,
                           'coarse_shift_y_px':0.,'coarse_shift_x_px':-20.,'coarse_pull_matrix_xy':matrix.tolist()}}
    fake_core(monkeypatch,checked)
    output=tmp_path/'composed-masked-identity'
    manifest=pipeline.prepare(path,output,{'frames':[0,1,2],'reference_frame':0,
        'registration':{'regions':[[0,0,60,64]],'exclude_regions':[[0,0,20,64]],'paths':{'2':[1]}}})
    support,_=read_sequence(output/manifest['correspondence_file'])
    assert support[1,:,20:60].all()
    assert support[2,:,20:40].all()
    assert not support[2,:,40:].any() and not support[2,:,:20].any()
    edge=next(row for row in manifest['registration_edges'] if row['fixed_source_frame']==1)
    assert edge['identity_control'] is True and edge['estimated'] is False
    with np.load(output/edge['support_file'],allow_pickle=False) as arrays:
        assert not arrays['correspondence_support'][:,:20].any()
        assert not arrays['correspondence_support'][:,60:].any()

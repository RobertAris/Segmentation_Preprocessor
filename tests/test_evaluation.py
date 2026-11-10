"""Known geometry checks for independent reference-space evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import dsa_preprocessor.evaluation as evaluation_module

from dsa_preprocessor.evaluation import annotation_template,evaluate_landmarks
from dsa_preprocessor.io import write_sequence
from dsa_preprocessor.transforms import affine_to_pull,save_pull_field


def prepared(tmp_path,matrix=None,support=True,fitting=None,valid=True):
    root=tmp_path/'prepared';root.mkdir()
    shape=(64,80)
    header={'space':'left-posterior-superior','space directions':np.eye(3),
            'space origin':[0,0,0],'kinds':['domain']*3}
    field=affine_to_pull(shape,np.eye(3) if matrix is None else matrix)
    source=tmp_path/'source_angio.nrrd'
    original=np.arange(np.prod(shape),dtype=np.uint16).reshape(shape)
    originals=np.stack([original,original+1])
    write_sequence(source,originals,header,[0,1])
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    frame_hashes=[hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in originals]
    metadata={'source_frame':1,'reference_frame':0,'source_sha256':source_hash}
    digest=save_pull_field(root/'transforms/frame-000001-pull.npz',field,metadata)
    write_sequence(root/'validity.seq.nrrd',np.ones((2,*shape),np.uint8),header,[0,1])
    support_array=np.ones((2,*shape),np.uint8)
    if not support:support_array[1]=0
    write_sequence(root/'correspondence.seq.nrrd',support_array,header,[0,1])
    manifest={'schema_version':2,'reference_frame':0,'source_sha256':source_hash,'source_file':str(source),
              'source_frame_count':2,
              'geometry':{'space':'left-posterior-superior','space directions':np.eye(3).tolist(),'space origin':[0,0,0]},
              'spatial_shape':list(shape),'validity_file':'validity.seq.nrrd',
              'correspondence_file':'correspondence.seq.nrrd','config':{'registration':{'fitting_landmarks':fitting or {}}},
              'frames':[{'source_frame':0,'registration_valid':True,'source_frame_sha256':frame_hashes[0]},
                        {'source_frame':1,'registration_valid':valid,'source_frame_sha256':frame_hashes[1],
                         'field_file':'transforms/frame-000001-pull.npz','field_sha256':digest}]}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root


def point(**kwargs):
    return {'source_frame':1,'point_id':'junction','role':'evaluation','visibility':'shared',
            'reference':[30.,24.],'moving':[30.,24.],'reference_width_px':4.,'region':'small-distal',**kwargs}


def packet(root,*rows):
    result=annotation_template(root)
    result['landmarks']=list(rows)
    return result


def test_missing_annotations_do_not_establish_zero_error_or_acceptance(tmp_path):
    root=prepared(tmp_path)
    result=evaluate_landmarks(root,packet(root))
    assert result['status']=='requires_annotations'
    assert result['summary']['reference_error_px']['median'] is None
    assert result['summary']['quantitative_targets_passed'] is False
    assert result['anatomical_acceptance_established'] is False


def test_inverse_error_is_measured_in_annotation_plane_under_compression(tmp_path):
    matrix=np.array([[.5,0,2],[0,.5,2],[0,0,1.]])
    root=prepared(tmp_path,matrix)
    result=evaluate_landmarks(root,packet(root,point(moving=[17.5,14.])))
    row=result['landmarks'][0]
    assert row['source_pull_residual_px']==pytest.approx(.5)
    assert row['reference_error_px']==pytest.approx(1.)
    assert row['reference_error_over_width']==pytest.approx(.25)
    assert result['summary']['quantitative_targets_passed'] is False


def test_fitting_points_are_excluded_and_cannot_be_relabelled_as_evaluation(tmp_path):
    fitting={'1':{'reference':[[30.,24.]],'moving':[[30.,24.]]}}
    root=prepared(tmp_path,fitting=fitting)
    result=evaluate_landmarks(root,packet(root,point(role='fitting')))
    assert result['summary']['shared_evaluation_landmarks']==0
    assert result['measurement_status_counts']=={'excluded_fitting':1}
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_landmarks(root,packet(root,point()))


def test_unobserved_branch_cannot_be_scored_as_wrong_or_correct_match(tmp_path):
    root=prepared(tmp_path)
    result=evaluate_landmarks(root,packet(root,point(visibility='reference_only',moving=None)))
    assert result['summary']['measured_landmarks']==0
    assert result['landmarks'][0]['reference_error_px'] is None


def test_unsupported_correspondence_fails_acceptance_even_with_zero_point_error(tmp_path):
    root=prepared(tmp_path,support=False)
    result=evaluate_landmarks(root,packet(root,point()))
    assert result['landmarks'][0]['reference_error_px']==pytest.approx(0.)
    assert result['landmarks'][0]['eligible_for_acceptance'] is False
    assert result['summary']['unsupported_or_failed_landmarks']==1
    assert result['summary']['quantitative_targets_passed'] is False


def test_folded_field_has_no_accepted_inverse(tmp_path):
    matrix=np.array([[-1.,0,79],[0,1.,0],[0,0,1.]])
    root=prepared(tmp_path,matrix)
    result=evaluate_landmarks(root,packet(root,point(moving=[49.,24.])))
    assert result['landmarks'][0]['reference_error_px'] is None
    assert result['landmarks'][0]['eligible_for_acceptance'] is False


def test_width_is_required_for_width_target_but_not_raw_error(tmp_path):
    root=prepared(tmp_path)
    result=evaluate_landmarks(root,packet(root,point(reference_width_px=None)))
    assert result['landmarks'][0]['reference_error_px']==0
    assert result['summary']['landmarks_without_reference_width']==1
    assert result['summary']['quantitative_targets_passed'] is False


def test_packet_fingerprint_and_field_hash_prevent_mixed_preparations(tmp_path):
    root=prepared(tmp_path)
    annotations=packet(root,point());annotations['prepared_manifest_sha256']='other'
    with pytest.raises(ValueError,match='fingerprint'):
        evaluate_landmarks(root,annotations)
    annotations=packet(root,point())
    target=root/'transforms/frame-000001-pull.npz'
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        evaluate_landmarks(root,annotations)


def test_fitting_or_ambiguous_points_do_not_inflate_success_statistics(tmp_path):
    root=prepared(tmp_path)
    annotations=packet(root,point(role='fitting'),point(point_id='unknown',visibility='ambiguous',reference=None,moving=None))
    result=evaluate_landmarks(root,annotations)
    assert result['summary']['shared_evaluation_landmarks']==0
    assert result['summary']['quantitative_targets_passed'] is False


def test_legacy_translation_can_be_evaluated_against_same_original_coordinates(tmp_path):
    root=prepared(tmp_path)
    path=root/'manifest.json';manifest=json.loads(path.read_text())
    manifest['schema_version']=1;manifest['config']={}
    manifest['frames'][1]={'source_frame':1,'status':'estimated','shift_y_px':-2.,'shift_x_px':3.}
    path.write_text(json.dumps(manifest))
    result=evaluate_landmarks(root,packet(root,point(moving=[27.,26.])))
    assert result['landmarks'][0]['reference_error_px']==pytest.approx(0.)
    assert result['summary']['quantitative_targets_passed'] is True
    assert result['anatomical_acceptance_established'] is False


def test_legacy_landmark_override_is_also_fit_data_not_independent_evidence(tmp_path):
    root=prepared(tmp_path)
    path=root/'manifest.json';manifest=json.loads(path.read_text())
    manifest['schema_version']=1
    manifest['config']={'landmarks':{'1':{'reference':[[30.,24.]],'moving':[[30.,24.]]}}}
    manifest['frames'][1]={'source_frame':1,'status':'manual','shift_y_px':0.,'shift_x_px':0.}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_landmarks(root,packet(root,point()))


def test_reference_hash_duplicate_is_identity_even_if_first_stored_copy(tmp_path):
    root=prepared(tmp_path)
    path=root/'manifest.json';manifest=json.loads(path.read_text())
    for row in manifest['frames']:
        row['source_frame_sha256']=hashlib.sha256(b'same-raw-frame').hexdigest()
    manifest['frames'][1]['independent_observation']=True
    path.write_text(json.dumps(manifest))
    result=evaluate_landmarks(root,packet(root,point()))
    assert result['landmarks'][0]['identity_control'] is True
    assert result['summary']['shared_evaluation_landmarks']==0
    assert result['summary']['quantitative_targets_passed'] is False


@pytest.mark.parametrize('mutation',['indices','geometry','binary'])
def test_support_sequence_metadata_and_binary_values_are_verified(tmp_path,mutation):
    root=prepared(tmp_path)
    header={'space':'left-posterior-superior','space directions':np.eye(3),
            'space origin':[0,0,0],'kinds':['domain']*3}
    masks=np.ones((2,64,80),np.uint8)
    indices=[0,1]
    if mutation=='indices':indices=[1,0]
    if mutation=='geometry':header['space origin']=[3,0,0]
    if mutation=='binary':masks[1,10,10]=2
    write_sequence(root/'correspondence.seq.nrrd',masks,header,indices)
    with pytest.raises(ValueError,match='index labels|geometry|binary'):
        evaluate_landmarks(root,packet(root,point()))


def test_tiny_fractional_weight_on_invalid_support_is_still_unsupported(tmp_path):
    root=prepared(tmp_path)
    header={'space':'left-posterior-superior','space directions':np.eye(3),
            'space origin':[0,0,0],'kinds':['domain']*3}
    masks=np.ones((2,64,80),np.uint8)
    masks[1,24,31]=0
    write_sequence(root/'correspondence.seq.nrrd',masks,header,[0,1])
    xy=[30.+1e-9,24.]
    result=evaluate_landmarks(root,packet(root,point(reference=xy,moving=xy)))
    assert result['landmarks'][0]['eligible_for_acceptance'] is False
    assert result['summary']['quantitative_targets_passed'] is False


def legacy_without_identity_metadata(root):
    path=root/'manifest.json';manifest=json.loads(path.read_text())
    manifest['schema_version']=1;manifest['config']={}
    manifest['frames']=[{'source_frame':0,'status':'reference','shift_y_px':0.,'shift_x_px':0.},
                        {'source_frame':1,'status':'estimated','shift_y_px':0.,'shift_x_px':0.}]
    path.write_text(json.dumps(manifest))
    return manifest


def test_legacy_nonduplicate_identity_is_verified_from_the_original_source(tmp_path):
    root=prepared(tmp_path)
    manifest=legacy_without_identity_metadata(root)
    result=evaluate_landmarks(root,packet(root,point()))
    assert result['landmarks'][0]['identity_control'] is False
    assert result['summary']['shared_evaluation_landmarks']==1
    provenance=result['identity_control_verification']
    assert provenance['methods_by_source_frame']['1']=='verified_original_source'
    assert provenance['verified_original_source_sha256']==manifest['source_sha256']
    assert result['evaluator_source_sha256']==hashlib.sha256(Path(evaluation_module.__file__).read_bytes()).hexdigest()


def test_legacy_earlier_reference_copy_is_excluded_using_original_frame_indices(tmp_path):
    root=prepared(tmp_path)
    manifest=legacy_without_identity_metadata(root)
    shape=tuple(manifest['spatial_shape'])
    original=np.arange(np.prod(shape),dtype=np.uint16).reshape(shape)
    header={'space':'left-posterior-superior','space directions':np.eye(3),
            'space origin':[0,0,0],'kinds':['domain']*3}
    source=Path(manifest['source_file'])
    write_sequence(source,np.stack([original,original+1,original]),header,[0,1,2])
    manifest.update(reference_frame=2,source_frame_count=3,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    manifest['frames']=[{'source_frame':0,'status':'estimated','shift_y_px':0.,'shift_x_px':0.},
                        {'source_frame':2,'status':'reference','shift_y_px':0.,'shift_x_px':0.}]
    (root/'manifest.json').write_text(json.dumps(manifest))
    write_sequence(root/'validity.seq.nrrd',np.ones((2,*shape),np.uint8),header,[0,2])
    result=evaluate_landmarks(root,packet(root,point(source_frame=0)))
    assert result['landmarks'][0]['identity_control'] is True
    assert result['summary']['shared_evaluation_landmarks']==0
    assert result['summary']['quantitative_targets_passed'] is False
    assert result['identity_control_verification']['methods_by_source_frame']['0']=='verified_original_source'


@pytest.mark.parametrize('problem',['missing','missing_field','hash_mismatch'])
def test_legacy_missing_identity_evidence_requires_available_matching_original_source(tmp_path,problem):
    root=prepared(tmp_path)
    manifest=legacy_without_identity_metadata(root)
    if problem=='missing':manifest['source_file']=str(tmp_path/'unavailable-original.nrrd')
    elif problem=='missing_field':manifest.pop('source_file')
    else:manifest['source_sha256']='0'*64
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='unavailable|source_file|SHA-256 mismatch'):
        evaluate_landmarks(root,packet(root,point()))


def test_schema2_frame_fingerprints_keep_evaluation_portable_without_original_source(tmp_path):
    root=prepared(tmp_path)
    path=root/'manifest.json';manifest=json.loads(path.read_text())
    manifest['source_file']=str(tmp_path/'unavailable-original.nrrd')
    path.write_text(json.dumps(manifest))
    result=evaluate_landmarks(root,packet(root,point()))
    assert result['summary']['shared_evaluation_landmarks']==1
    provenance=result['identity_control_verification']
    assert provenance['methods_by_source_frame']['1']=='persisted_frame_sha256'
    assert provenance['verified_original_source_sha256'] is None

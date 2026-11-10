"""Dense review artifacts and Slicer-independent annotation eligibility checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types

import numpy as np
from PIL import Image
import pytest

from dsa_preprocessor.io import write_sequence
from dsa_preprocessor.report import write_report


@pytest.fixture
def dense_run(tmp_path):
    height,width = 40,720
    yy,xx = np.mgrid[:height,:width]
    reference = (1000-100*np.exp(-((yy-20)**2+(xx-180)**2)/35)).astype(np.float32)
    originals = np.stack([reference,reference])
    valid = np.ones_like(originals,dtype=bool)
    valid[1,:2] = False
    support = np.ones_like(valid)
    support[1,:,width//2:] = False
    confidence = support.astype(np.float32)*.75
    header = {'space':'left-posterior-superior','space directions':np.diag([.2,.3,1.]),
              'space origin':[3.,4.,5.],'kinds':['domain']*3}
    write_sequence(tmp_path/'correspondence.seq.nrrd',support.astype(np.uint8),header,[2,9])
    write_sequence(tmp_path/'confidence.seq.nrrd',confidence,header,[2,9])
    frames=[]
    (tmp_path/'transforms').mkdir()
    for source,status in [(2,'reference'),(9,'review')]:
        path=tmp_path/'transforms'/f'frame-{source:06d}-pull.npz'
        np.savez_compressed(path,pull_displacement_yx=np.zeros((2,height,width),dtype=np.float32))
        frames.append({'source_frame':source,'status':status,'registration_valid':True,
                       'annotation_eligible':source==2,'field_file':str(path.relative_to(tmp_path)),
                       'field_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'coarse_shift_y_px':0.,'coarse_shift_x_px':0.,
                       'coarse_pull_matrix_xy':np.eye(3).tolist(),'ncc_before':1.,'ncc_after':1.,
                       'qa':{'min_jacobian':1.,'folded_cell_fraction':0.},'included_in_minip':True})
    manifest={'schema_version':2,'run_id':'test-dense','source_file':'source_angio.nrrd',
              'reference_frame':2,'spatial_shape':[height,width],'frames':frames,
              'group_id':'test-dense:reference-2','registration_method':'bspline',
              'config':{'reference_frame':2,'frames':[2,9]},
              'correspondence_file':'correspondence.seq.nrrd','confidence_file':'confidence.seq.nrrd'}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return tmp_path,manifest,originals,valid


def test_dense_report_has_native_resolution_and_correspondence_evidence(dense_run):
    directory,manifest,originals,valid = dense_run
    report=write_report(directory,manifest,originals,originals,valid)
    html=report.read_text()
    packet=json.loads(re.search(r'<script id="frame-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    editor=json.loads(re.search(r'<script id="editor-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert [p['source'] for p in packet]==[2,9]
    assert editor['manifest_sha256']==hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()
    assert Image.open(directory/packet[1]['full_urls']['original']).size==(720,40)
    assert Image.open(directory/packet[1]['urls']['original']).size[0]==640
    assert {'support','confidence','coarse','displacement','deformation'}<=set(packet[1]['extra_urls'])
    support=np.asarray(Image.open(directory/packet[1]['extra_urls']['support']))
    np.testing.assert_array_equal(support[10,600],[235,115,25])
    np.testing.assert_array_equal(support[0,100],[235,115,25])
    assert 'original_full_resolution_xy' in html
    assert 'prepared_manifest_sha256' in html
    assert 'Coarse dx' in html
    assert 'source_frame:f.source' in html
    assert 'Changing the reference clears manual shifts' in html
    for frame in packet:
        for collection in ['urls','full_urls','extra_urls']:
            assert all((directory/url).is_file() for url in frame[collection].values())


def test_dense_report_refuses_changed_transform(dense_run):
    directory,manifest,originals,valid=dense_run
    path=directory/manifest['frames'][1]['field_file']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='fingerprint'):
        write_report(directory,manifest,originals,originals,valid)


@pytest.mark.parametrize('key,bad_value,expected', [('correspondence_file',2.,'binary'),
                                                 ('confidence_file',1.5,'Confidence')])
def test_dense_report_rejects_invalid_screening_maps(dense_run,key,bad_value,expected):
    from dsa_preprocessor.io import read_sequence
    directory,manifest,originals,valid=dense_run
    arrays,header=read_sequence(directory/manifest[key])
    arrays=arrays.astype(np.float32);arrays[1,3,4]=bad_value
    write_sequence(directory/manifest[key],arrays,header,[2,9])
    with pytest.raises(ValueError,match=expected):
        write_report(directory,manifest,originals,originals,valid)


def test_legacy_report_remains_translation_review(tmp_path):
    image=np.arange(30*40,dtype=np.float32).reshape(1,30,40)
    manifest={'schema_version':1,'run_id':'legacy','reference_frame':7,
              'frames':[{'source_frame':7,'status':'reference','shift_y_px':0.,'shift_x_px':0.}],
              'config':{'reference_frame':7,'frames':[7]}}
    report=write_report(tmp_path,manifest,image,image,np.ones_like(image,dtype=bool))
    html=report.read_text()
    assert 'Draw registration ROI' in html
    assert 'Download updated JSON' in html
    assert 'id="download-decisions"' not in html
    assert (tmp_path/'review/frame-0000-original.png').is_file()


@pytest.fixture
def policy(monkeypatch):
    # Import only pure policy helpers; no MRML or GUI behavior is simulated here.
    for name in ['qt','slicer','vtk']:
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    path=Path(__file__).resolve().parents[1]/'slicer/DSAWorkflow.py'
    spec=importlib.util.spec_from_file_location('dsa_workflow_policy_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_manifest():
    return {'schema_version':2,'spatial_shape':[20,30],'group_id':'g',
            'frames':[{'source_frame':2,'status':'reference','registration_valid':True},
                      {'source_frame':9,'status':'review','registration_valid':True},
                      {'source_frame':12,'status':'unsupported','registration_valid':False}]}


def test_failed_candidate_cannot_be_approved_by_review_file(policy):
    record={'schema_version':1,'prepared_manifest_sha256':'hash','group_id':'g',
            'decisions':[{'source_frame':12,'decision':'accepted','regions':[]}]}
    with pytest.raises(ValueError,match='Cannot approve'):
        policy._validated_review_record(record,policy_manifest(),'hash')
    record['decisions'][0]={'source_frame':9,'decision':'accepted','regions':[[1,2,4,5]],'notes':'shared branches checked'}
    assert policy._validated_review_record(record,policy_manifest(),'hash')[9]['regions']==[[1,2,4,5]]
    with pytest.raises(ValueError,match='different preparation'):
        policy._validated_review_record(record,policy_manifest(),'different-hash')


@pytest.mark.parametrize('regions', [[[29,0,2,2]], [[1.5,2,3,4]], [[1,2,0,4]], [1,2,3,4]])
def test_review_regions_cannot_claim_invalid_reference_coverage(policy,regions):
    with pytest.raises(ValueError):
        policy._validated_regions(regions,[20,30])


def test_identity_and_frame_review_do_not_imply_full_tree_coverage(policy):
    full=np.ones((20,30),dtype=bool)
    assert not policy._reviewed_coverage((20,30),{}, {2:full}).any()
    assert not policy._reviewed_coverage((20,30),{2:[]}, {2:full}).any()
    support=full.copy();support[:,7:]=False
    coverage=policy._reviewed_coverage((20,30),{9:[[3,4,8,6]]},{9:support})
    assert coverage.sum()==4*6
    assert coverage[4:10,3:7].all()
    assert not coverage[:,7:].any()


def test_review_cannot_override_folding_or_pending_edits(policy):
    manifest=policy_manifest()
    with pytest.raises(RuntimeError,match='Explicit anatomical review'):
        policy._reviewed_contributors(manifest,{9},set())
    assert policy._reviewed_contributors(manifest,{9},{9})==[9]
    manifest['frames'][1]['qa']={'folded_cell_fraction':.001}
    with pytest.raises(RuntimeError,match='ineligible'):
        policy._reviewed_contributors(manifest,{9},{9})
    with pytest.raises(RuntimeError,match='override'):
        policy._reviewed_contributors(manifest,set(),{12})


def test_legacy_eligibility_policy_still_blocks_rejected_edits(policy):
    manifest={'schema_version':1,'frames':[{'source_frame':1,'status':'estimated'},
                                         {'source_frame':2,'status':'rejected'}]}
    assert policy._reviewed_contributors(manifest,{1},set())==[1]
    with pytest.raises(RuntimeError,match='ineligible'):
        policy._reviewed_contributors(manifest,{2},set())


def test_export_representation_events_do_not_become_false_paint_provenance(policy):
    session=object.__new__(policy.DSASession)
    session._initializing=False
    session.edited_source_frames=set()
    session.source_frames=[9]
    session.browser=types.SimpleNamespace(GetSelectedItemNumber=lambda:0)
    def serialize(*args):
        session._on_segment_change()
        raise RuntimeError('save failed')
    session._export_snapshot=serialize
    with pytest.raises(RuntimeError,match='save failed'):
        session.export()
    assert session.edited_source_frames==set()
    assert session._exporting is False
    session._on_segment_change()
    assert session.edited_source_frames=={9}

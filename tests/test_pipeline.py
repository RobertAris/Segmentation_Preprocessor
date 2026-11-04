"""Integration check: selected indices, manual transforms and source preservation."""
import csv
import hashlib
import json

import nrrd
import numpy as np
import pytest
from scipy.ndimage import shift

from dsa_preprocessor import pipeline
from dsa_preprocessor.pipeline import prepare


@pytest.mark.parametrize('shift_key', ['1', 1])
def test_noncontiguous_source_indices_and_manual_override(tmp_path, shift_key):
    yy, xx = np.mgrid[:48, :64]
    reference = (1200 - 500*np.exp(-((xx-25)**2/3 + (yy-22)**2/60))).astype(np.uint16)
    source = np.stack([reference, shift(reference, (2, -3), order=0, mode='nearest'),
                       reference, reference])
    path = tmp_path / 'sample_angio.nrrd'
    header = {
        'space': 'left-posterior-superior',
        'space directions': np.diag([.2, .3, 1.]),
        'space origin': np.array([2., 4., 8.]),
        'kinds': ['domain', 'domain', 'domain'],
    }
    nrrd.write(str(path), source.transpose(2, 1, 0), header, index_order='F')
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / 'prepared'
    manifest = prepare(path, output, {
        'frames': [1, 3], 'reference_frame': 3,
        'manual_shifts': {shift_key: [-2, 3]},
        'max_shift_px': 12,
    })
    assert manifest['reference_frame'] == 3
    assert [frame['source_frame'] for frame in manifest['frames']] == [1, 3]
    np.testing.assert_allclose(
        [manifest['frames'][0]['shift_y_px'], manifest['frames'][0]['shift_x_px']], [-2, 3])
    np.testing.assert_allclose(
        [manifest['frames'][1]['shift_y_px'], manifest['frames'][1]['shift_x_px']], [0, 0])
    data, metadata = nrrd.read(str(output / manifest['sequence_file']), index_order='F')
    assert data.shape == (64, 48, 1, 2)
    assert metadata['axis 3 index values'] == '1 3'
    np.testing.assert_array_equal(data[8:-8, 8:-8, 0, 0], data[8:-8, 8:-8, 0, 1])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    saved = json.loads((output / 'manifest.json').read_text())
    assert saved['reference_frame'] == 3


@pytest.fixture
def dataset_source(tmp_path):
    root = tmp_path / 'source-dataset'
    image_dir = root / 'sub-demo' / 'xa'
    image_dir.mkdir(parents=True)
    path = image_dir / 'sub-demo_acq-hepatic_run-1_angio.nrrd'
    yy, xx = np.mgrid[:48, :64]
    pattern = (1200 - 500*np.exp(-((xx-25)**2/3 + (yy-22)**2/60))).astype(np.uint16)
    frames = np.stack([pattern, pattern, pattern, pattern])
    header = {
        'space': 'left-posterior-superior',
        'space directions': np.diag([.2, .3, 1.]),
        'space origin': [0., 0., 0.], 'kinds': ['domain']*3,
    }
    nrrd.write(str(path), frames.transpose(2, 1, 0), header, index_order='F')
    path.with_suffix('.json').write_text(json.dumps({'PixelSpacingKnown': True, 'NumberOfFrames': 4}))
    with (root / 'series.tsv').open('w', newline='') as stream:
        fields = ['participant_id', 'acq', 'run', 'suffix', 'note']
        writer = csv.DictWriter(stream, fields, delimiter='\t')
        writer.writeheader()
        writer.writerow({'participant_id': 'sub-demo', 'acq': 'hepatic', 'run': 1,
                         'suffix': 'angio',
                         'note': 'Last frame is made by the scanner. It is a summary picture of the whole run.'})
    return path


def test_summary_frame_from_series_table_is_not_registered_by_default(dataset_source, tmp_path):
    entry = pipeline.inspect_dataset(dataset_source)[0]
    assert entry['summary_frame'] == 3
    manifest = prepare(dataset_source, tmp_path / 'default', {'reference_frame': 1})
    assert [frame['source_frame'] for frame in manifest['frames']] == [0, 1, 2]
    assert manifest['summary_frame'] == 3
    with pytest.raises(ValueError, match='summary'):
        prepare(dataset_source, tmp_path / 'explicit', {'frames': [1, 3], 'reference_frame': 1})
    assert not (tmp_path / 'explicit').exists()


def test_auto_reference_uses_original_indices_after_frame_selection(dataset_source, tmp_path, monkeypatch):
    # The selection scorer returns one value per ORIGINAL frame; frame 3 is a
    # summary and must not win even with the largest score.
    monkeypatch.setattr(pipeline, '_frame_scores', lambda *args, **kwargs: [1., 2., 8., 99.])
    manifest = prepare(dataset_source, tmp_path / 'automatic', {'frames': [0, 2]})
    assert manifest['reference_frame'] == 2
    assert manifest['reference_selection'] == 'automatic'
    assert [frame['source_frame'] for frame in manifest['frames']] == [0, 2]
    assert manifest['frames'][1]['status'] == 'reference'
    assert manifest['frames'][1]['shift_y_px'] == manifest['frames'][1]['shift_x_px'] == 0


def test_unknown_configuration_is_rejected_without_partial_output(dataset_source, tmp_path):
    target = tmp_path / 'unknown'
    with pytest.raises(ValueError, match='Unknown configuration'):
        prepare(dataset_source, target, {'referance_frame': 2})
    assert not target.exists()


def test_existing_output_is_preserved_without_overwrite(dataset_source, tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    marker = target / 'annotation.seg.nrrd'
    marker.write_bytes(b'irreplaceable annotation')
    with pytest.raises(ValueError, match='already exists'):
        prepare(dataset_source, target)
    assert marker.read_bytes() == b'irreplaceable annotation'
    assert list(target.iterdir()) == [marker]


def test_source_dataset_cannot_be_used_as_derivative_destination(dataset_source):
    target = dataset_source.parent / 'prepared'
    with pytest.raises(ValueError, match='outside the source dataset'):
        prepare(dataset_source, target)
    assert not target.exists()


@pytest.mark.parametrize('config', [
    {'frames': [0, 0]},
    {'frames': [0, 1.5]},
    {'frames': [0, 2], 'reference_frame': 1},
    {'frames': [0, 2], 'manual_shifts': {'1': [0, 0]}},
    {'frames': [0, 2], 'reference_frame': 2, 'manual_shifts': {'2': [1, 0]}},
])
def test_invalid_frame_selection_and_reference_geometry_are_rejected(dataset_source, tmp_path, config):
    target = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        prepare(dataset_source, target, config)
    assert not target.exists()

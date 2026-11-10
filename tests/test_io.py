"""Check temporal and spatial axes separately on non-square images."""
import numpy as np
import nrrd

from dsa_preprocessor.io import read_sequence, write_sequence, write_volume


def geometry():
    return {
        'space': 'left-posterior-superior',
        'space directions': np.array([[0., .2, 0.], [-.3, 0., 0.], [0., 0., 1.]]),
        'space origin': np.array([17., -4., 2.]),
        'kinds': ['domain', 'domain', 'domain'],
    }


def test_reads_legacy_xy_time_as_time_y_x(tmp_path):
    frames = np.arange(3*5*7, dtype=np.uint16).reshape(3, 5, 7)
    path = tmp_path / 'legacy_angio.nrrd'
    nrrd.write(str(path), frames.transpose(2, 1, 0), geometry(), index_order='F')
    actual, header = read_sequence(path)
    np.testing.assert_array_equal(actual, frames)
    np.testing.assert_allclose(header['space origin'], geometry()['space origin'])


def test_written_sequence_is_real_slicer_four_dimensional_sequence(tmp_path):
    frames = np.arange(3*5*7, dtype=np.uint16).reshape(3, 5, 7)
    path = tmp_path / 'registered.seq.nrrd'
    write_sequence(path, frames, geometry(), [2, 5, 9])
    data, header = nrrd.read(str(path), index_order='F')
    assert data.shape == (7, 5, 1, 3)
    assert header['kinds'] == ['domain', 'domain', 'domain', 'list']
    assert header['axis 3 index values'] == '2 5 9'
    assert header['DataNodeClassName'] == 'vtkMRMLScalarVolumeNode'
    np.testing.assert_array_equal(data[:, :, 0, :], frames.transpose(2, 1, 0))
    restored, _ = read_sequence(path)
    np.testing.assert_array_equal(restored, frames)
    np.testing.assert_allclose(header['space directions'][:3], geometry()['space directions'])
    assert np.isnan(header['space directions'][3]).all()
    np.testing.assert_allclose(header['space origin'], geometry()['space origin'])


def test_volume_has_one_real_slice_and_original_in_plane_geometry(tmp_path):
    image = np.arange(5*7, dtype=np.float32).reshape(5, 7)
    path = tmp_path / 'reference.nrrd'
    write_volume(path, image, geometry())
    data, header = nrrd.read(str(path), index_order='F')
    assert data.shape == (7, 5, 1)
    np.testing.assert_array_equal(data[:, :, 0], image.T)
    np.testing.assert_allclose(header['space directions'][:2], geometry()['space directions'][:2])
    np.testing.assert_allclose(header['space origin'], geometry()['space origin'])

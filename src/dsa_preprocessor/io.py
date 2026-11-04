"""NRRD I/O. Public arrays are always (time, row, column), never spatial z=time."""
from pathlib import Path

import numpy as np
import nrrd


def read_sequence(path):
    data, header = nrrd.read(str(path), index_order="C")
    if data.ndim == 4:
        kinds = list(header.get("kinds", []))
        if data.shape[1] != 1 or (kinds and kinds[-1] != "list"):
            raise ValueError("Expected a 2D sequence with singleton spatial z and last NRRD axis list")
        data = data[:, 0]
    elif data.ndim == 2:
        data = data[None]
    elif data.ndim != 3:
        raise ValueError("Expected a 2D+time NRRD (source x,y,time or prepared x,y,1,list)")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise ValueError("Input must contain finite numeric intensities")
    spatial_header(header)
    return data, header


def spatial_header(header):
    """Retain in-plane geometry; the source's third index is time, not a z offset."""
    directions = np.asarray(header.get("space directions", np.eye(3)), dtype=float)
    if directions.shape[0] < 2 or directions.shape[1:] != (3,):
        raise ValueError("A 3D physical coordinate system with two in-plane axes is required")
    x, y = directions[:2]
    normal = np.cross(x, y)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
        raise ValueError("Invalid or parallel in-plane directions")
    normal /= np.linalg.norm(normal)
    origin = np.asarray(header.get("space origin", [0, 0, 0]), dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid space origin")
    result = {"space": header.get("space", "left-posterior-superior"),
              "space directions": np.stack([x, y, normal]),
              "space origin": origin, "kinds": ["domain"] * 3,
              "encoding": "gzip"}
    if "space units" in header:
        result["space units"] = header["space units"]
    return result


def write_volume(path, array, header):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError("A DSA reference volume must have exactly one spatial slice")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(path), array, header=spatial_header(header), index_order="C", compression_level=1)


def write_sequence(path, array, header, indices=None):
    array = np.asarray(array)
    if array.ndim != 3 or not array.shape[0]:
        raise ValueError("Sequence array must have shape (time, y, x)")
    if indices is None:
        indices = list(range(array.shape[0]))
    if len(indices) != len(array) or len(set(map(str, indices))) != len(indices):
        raise ValueError("Sequence indices must be unique and match the number of frames")
    result = spatial_header(header)
    result["space directions"] = np.vstack([result["space directions"], [np.nan] * 3])
    result["kinds"] = ["domain", "domain", "domain", "list"]
    result["labels"] = ["", "", "", "source_frame"]
    result["axis 3 index type"] = "numeric"
    result["axis 3 index values"] = " ".join(map(str, indices))
    result["DataNodeClassName"] = "vtkMRMLScalarVolumeNode"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(path), array[:, None], header=result, index_order="C", compression_level=1)

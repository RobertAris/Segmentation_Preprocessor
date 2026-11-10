"""Projection values, support distinctions, atomic output and checkout provenance."""
import hashlib
import json
from pathlib import Path
import subprocess

import nrrd
import numpy as np
from PIL import Image
import pytest
from scipy.ndimage import shift

from dsa_preprocessor import pipeline
from dsa_preprocessor.io import read_sequence
from dsa_preprocessor.projections import write_projection_outputs


HEADER = {"space": "left-posterior-superior",
          "space directions": [[0., .2, 0.], [-.3, 0., 0.], [0., 0., 1.]],
          "space origin": [2., 4., 8.], "kinds": ["domain"]*3}


@pytest.mark.parametrize("bright", [False, True])
def test_all_frame_projection_omits_invalid_samples_but_retains_excluded_frames(tmp_path, bright):
    originals = np.full((3, 8, 10), 100., dtype=np.float32)
    originals[1, 3, 4], originals[2, 4, 6] = 5., 9.
    aligned = originals.copy()
    aligned[1, 3, 4], aligned[1, 3, 3] = 100., 5.
    aligned[1, 0, 0] = -1000.  # Invalid padding must not create a projected vessel.
    valid = np.ones_like(aligned, dtype=bool)
    valid[1, 0, 0] = False
    valid[:, 0, 1] = False
    curated = np.full((8, 10), 100., dtype=np.float32)
    if bright:
        originals, aligned, curated = -originals, -aligned, -curated
    manifest = {"schema_version": 2, "run_id": "synthetic <unsafe> title",
                "reference_frame": 2, "config": {"polarity": "bright" if bright else "dark"},
                "frames": [{"source_frame": 2, "included_in_minip": True},
                           {"source_frame": 8, "included_in_minip": False},
                           {"source_frame": 12, "included_in_minip": False}]}
    metadata = write_projection_outputs(tmp_path, manifest, originals, aligned, valid,
                                        curated, np.ones((8, 10), bool), HEADER)
    before, before_header = read_sequence(tmp_path/metadata["paths"]["before_nrrd"])
    after, after_header = read_sequence(tmp_path/metadata["paths"]["after_nrrd"])
    counts, _ = read_sequence(tmp_path/metadata["paths"]["after_frame_count_nrrd"])
    after_valid, _ = read_sequence(tmp_path/metadata["paths"]["after_validity_nrrd"])
    operation = np.max if bright else np.min
    np.testing.assert_array_equal(before[0], operation(originals, axis=0))
    expected = operation(np.where(valid, aligned, -np.inf if bright else np.inf), axis=0)
    np.testing.assert_array_equal(after[0][after_valid[0].astype(bool)], expected[after_valid[0].astype(bool)])
    assert counts[0, 0, 0] == 2 and counts[0, 0, 1] == 0
    assert not after_valid[0, 0, 1]
    assert after[0, 0, 0] == (-100 if bright else 100)
    assert after[0, 3, 3] == (-5 if bright else 5)
    assert after[0, 4, 6] == (-9 if bright else 9)
    assert after_header["kinds"] == before_header["kinds"] == ["domain"]*3
    np.testing.assert_array_equal(after_header["space directions"], HEADER["space directions"])
    np.testing.assert_array_equal(after_header["space origin"], HEADER["space origin"])
    assert metadata["source_frames"] == [2, 8, 12]
    assert metadata["included_in_comparison_but_excluded_from_curated"] == [8, 12]
    assert metadata["curated_source_frames"] == [2]
    assert metadata["operation"] == ("maximum" if bright else "minimum")
    assert metadata["label"] == ("MIP" if bright else "MinIP")
    assert metadata["anatomical_acceptance_established"] is False
    assert metadata["no_additional_resampling"] is True
    window = metadata["display_window"]
    expected_before = np.uint8(np.clip((before[0]-window["low"])/(window["high"]-window["low"]), 0, 1)*255)
    expected_after = np.uint8(np.clip((after[0]-window["low"])/(window["high"]-window["low"]), 0, 1)*255)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/metadata["paths"]["before_png"])), expected_before)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/metadata["paths"]["after_png"])), expected_after)
    html = (tmp_path/metadata["paths"]["viewer"]).read_text()
    assert "synthetic &lt;unsafe&gt; title" in html
    assert "<unsafe>" not in html
    assert "maximum over time" in html if bright else "minimum over time" in html
    assert "2, 8, 12" in html and "Start blinking" in html
    assert all((tmp_path/path).is_file() for path in metadata["paths"].values())


def test_projection_constant_single_frame_and_empty_curated_support(tmp_path):
    image = np.full((1, 4, 5), 23., dtype=np.float32)
    manifest = {"reference_frame": 7, "frames": [{"source_frame": 7, "included_in_minip": False}]}
    metadata = write_projection_outputs(tmp_path, manifest, image, image, np.ones_like(image, bool),
                                        image[0], np.zeros((4, 5), bool), HEADER)
    assert metadata["display_window"]["high"] > metadata["display_window"]["low"]
    assert metadata["frame_count"] == 1
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/"minip.png")), 255)
    before, _ = read_sequence(tmp_path/"minip-comparison/before.nrrd")
    after, _ = read_sequence(tmp_path/"minip-comparison/after.nrrd")
    np.testing.assert_array_equal(before, image)
    np.testing.assert_array_equal(after, image)


@pytest.mark.parametrize("problem", ["nonbinary", "duplicate", "nonfinite", "shape"])
def test_projection_rejects_inconsistent_inputs_before_publishing(tmp_path, problem):
    original = np.full((2, 8, 10), 100., dtype=np.float32)
    valid = np.ones_like(original, dtype=np.float32)
    manifest = {"reference_frame": 0, "frames": [{"source_frame": 0}, {"source_frame": 1}]}
    if problem == "nonbinary":
        valid[1, 3, 4] = .5
    elif problem == "duplicate":
        manifest["frames"][1]["source_frame"] = 0
    elif problem == "nonfinite":
        original[1, 3, 4] = np.nan
    else:
        valid = valid[:, :-1]
    with pytest.raises(ValueError):
        write_projection_outputs(tmp_path, manifest, original, original, valid,
                                 np.full((8, 10), 100.), np.ones((8, 10), bool), HEADER)
    assert not (tmp_path/"minip.png").exists()
    assert not (tmp_path/"minip-comparison").exists()


@pytest.fixture
def source(tmp_path):
    yy, xx = np.mgrid[:48, :64]
    image = (1200-500*np.exp(-((xx-25)**2/3+(yy-22)**2/60))).astype(np.uint16)
    images = np.stack([image, shift(image, (2, -3), order=0, mode="nearest"), image, image])
    path = tmp_path/"synthetic_angio.nrrd"
    nrrd.write(str(path), images, HEADER, index_order="C")
    return path, images


@pytest.mark.parametrize("schema", [1, 2])
def test_prepare_generates_projection_contract_and_preserves_curated_semantics(source, tmp_path, schema):
    path, images = source
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {"frames": [1, 3], "reference_frame": 3, "exclude_from_minip": [1]}
    if schema == 2:
        config["registration"] = {"method": "translation"}
    else:
        config["manual_shifts"] = {"1": [-2, 3]}
    output = tmp_path/"prepared"
    manifest = pipeline.prepare(path, output, config)
    assert manifest["schema_version"] == schema
    assert (output/"minip.png").is_file()
    before, _ = read_sequence(output/"minip-comparison/before.nrrd")
    np.testing.assert_array_equal(before[0], images[[1, 3]].min(axis=0))
    curated, _ = read_sequence(output/manifest["minip_file"])
    np.testing.assert_array_equal(curated[0], images[3])
    metadata = manifest["projection_comparison"]
    assert metadata["source_frames"] == [1, 3]
    assert metadata["included_in_comparison_but_excluded_from_curated"] == [1]
    assert metadata == json.loads((output/"minip-comparison/projection.json").read_text())
    assert manifest == json.loads((output/"manifest.json").read_text())
    assert "minip-comparison/index.html" in (output/"report.html").read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize("schema", [1, 2])
def test_projection_write_failure_rolls_back_entire_preparation(source, tmp_path, monkeypatch, schema):
    from dsa_preprocessor import projections
    path, _ = source
    output = tmp_path/"failed"
    config = {"frames": [0, 1], "reference_frame": 0}
    if schema == 2:
        config["registration"] = {"method": "translation"}
    real_write = projections.write_volume
    writes = []
    def broken(*args):
        writes.append(str(args[0]))
        if len(writes) == 2:
            raise OSError("projection disk failure")
        return real_write(*args)
    monkeypatch.setattr(projections, "write_volume", broken)
    with pytest.raises(OSError, match="projection disk failure"):
        pipeline.prepare(path, output, config)
    assert len(writes) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_batch_preparations_receive_automatic_projection_outputs(source, tmp_path):
    from dsa_preprocessor.cli import main
    path, _ = source
    output = tmp_path/"batch"
    assert main(["batch", str(path), "-o", str(output)]) == 0
    runs = [directory for directory in output.iterdir() if directory.is_dir()]
    assert len(runs) == 1
    assert (runs[0]/"minip.png").is_file()
    assert (runs[0]/"minip-comparison/index.html").is_file()
    manifest = json.loads((runs[0]/"manifest.json").read_text())
    assert manifest["projection_comparison"]["frame_count"] == 4


def test_installed_package_does_not_claim_consumer_repository_revision(tmp_path, monkeypatch):
    consumer = tmp_path/"consumer"
    installed = consumer/".venv/lib/python/site-packages/dsa_preprocessor"
    installed.mkdir(parents=True)
    (installed/"pipeline.py").write_text("# installed wheel\n")
    (installed/"__init__.py").write_text("__version__ = 'fixture'\n")
    subprocess.run(["git", "init", "-q", str(consumer)], check=True)
    (consumer/"consumer.txt").write_text("unrelated project\n")
    subprocess.run(["git", "add", "consumer.txt"], cwd=consumer, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "consumer"], cwd=consumer, check=True)
    monkeypatch.setattr(pipeline, "__file__", str(installed/"pipeline.py"))
    provenance = pipeline._code_provenance()
    assert provenance["code_revision"] is None
    assert provenance["software_version"] == pipeline.__version__
    assert provenance["source_code_sha256"]["pipeline.py"] == hashlib.sha256((installed/"pipeline.py").read_bytes()).hexdigest()


def test_tracked_package_checkout_records_its_own_revision(tmp_path, monkeypatch):
    checkout = tmp_path/"checkout"
    package = checkout/"src/dsa_preprocessor"
    package.mkdir(parents=True)
    (package/"pipeline.py").write_text("# source checkout\n")
    (package/"__init__.py").write_text("__version__ = 'fixture'\n")
    (checkout/"pyproject.toml").write_text('[project]\nname = "dsa-preprocessor"\n')
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "add", "src", "pyproject.toml"], cwd=checkout, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "package"], cwd=checkout, check=True)
    monkeypatch.setattr(pipeline, "__file__", str(package/"pipeline.py"))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    assert pipeline._code_provenance()["code_revision"] == revision

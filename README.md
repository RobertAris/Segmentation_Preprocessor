# DSA segmentation preprocessor

Align frames from a 2D digital subtraction angiography (DSA) sequence to one fixed
reference, inspect their minimum-intensity projection (MinIP), and accumulate
visible vessels into one static annotation in 3D Slicer.

The pipeline supports translation, rigid, similarity, affine, B-spline, TV-L1 and
optional diffeomorphic Demons registration. It resamples original intensities once,
retains original frame indices and reference geometry, and records transformations
and provenance. Local deformation remains a candidate until shared anatomy is
reviewed. Changing contrast, projected overlap and missing vessels can prevent
reliable correspondence; a sharper projection alone does not establish accuracy.

## Install

Python 3.10 or newer is required. From a checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
dsa-preprocess --help
```

For editable development, use `python -m pip install -e '.[test]'`. To enable
Demons, install `python -m pip install '.[registration]'`. Slicer is a separate
application; the annotation helper is included in this checkout's `slicer/` folder.
AI Lab is optional and is not required to install or run the preprocessor.

## Prepare a sequence

Supply a local NRRD whose third source axis is time. No acquisition images or
patient-specific configurations are included in the repository. Replace the input
path below with your sequence:

```sh
dsa-preprocess configure /path/to/sequence_angio.nrrd -o configs/my-run.json
```

Review the generated JSON: choose the actual reference frame, selected frames and
vessel regions. Frame indices start at **0**. Then prepare a new output directory:

```sh
dsa-preprocess prepare /path/to/sequence_angio.nrrd \
  --config configs/my-run.json --registration-method bspline \
  -o outputs/my-run
```

Omit `--registration-method` and the nested `registration` object for the legacy
translation workflow. The generic [translation](examples/translation.json) and
[B-spline](examples/bspline.json) configurations show minimal settings; acquisition
selection and regions still need review. The [local alignment guide](docs/deformable-alignment.md)
explains multiple regions, exclusions, landmarks and explicit keyframe paths.

Open `outputs/my-run/report.html` in a browser. On macOS:

```sh
open outputs/my-run/report.html
```

The report provides original/aligned frames, native-resolution overlays, blink,
quality diagnostics and local correspondence support. Check shared bifurcations
throughout the tree, including peripheral branches. Orange in the correspondence
view identifies unsupported matching. A confidence score is not anatomical approval.

Changes in the browser update a downloadable configuration. To apply them, rerun
`prepare` with that JSON and a **new** output directory. Existing output directories
are protected from replacement. Do not change the reference under an existing
annotation; prepare a separate group when projections are incompatible.

## Automatic MinIP comparison

**Every successful preparation automatically creates a projection and a before/after
viewer**, including runs created through `batch` or `pilot`. Open
`outputs/my-run/minip-comparison/index.html` to compare native images or blink
between them. No separate projection command is needed.

The comparison uses exactly the same selected frames before and after alignment,
excluding only invalid source sampling after alignment, with one linear display
window and no added sharpening or denoising. It includes failed/excluded frames
for diagnostic comparison, which are identified in the viewer. The curated
`minip.nrrd` excludes ineligible frames and, for local methods, unsupported matching.
Fewer contributions can reduce apparent blur, so these two projections have
explicitly different meanings.

For dark vessels the projection is a **minimum** (MinIP). For bright vessels,
`polarity: "bright"` produces a **maximum** (MIP), labeled accordingly in the viewer.
The historical `minip` filenames are retained for compatibility.

| File | Purpose |
| --- | --- |
| `registered.seq.nrrd` | Original intensities in the fixed reference grid, with a time axis |
| `reference.nrrd` | Single-slice annotation reference |
| `validity.seq.nrrd` | Valid original-image interpolation support |
| `minip.nrrd`, `minip_validity.nrrd`, `minip.png` | Curated projection and validity |
| `minip-comparison/index.html` | Matched-frame comparison and blink viewer |
| `minip-comparison/before.nrrd`, `after.nrrd` | Original/aligned all-selected-frame projections |
| `minip-comparison/before.png`, `after.png`, `comparison.png` | Native PNGs and combined view |
| `minip-comparison/after-validity.nrrd`, `after-frame-count.nrrd` | Sampling coverage and contributor counts |
| `minip-comparison/projection.json` | Projection settings and frame/support semantics |
| `manifest.json`, `config.json`, `transforms.tsv` | Source/configuration/transformation provenance |
| `report.html`, `review/` | Frame review and image assets |

Local methods additionally save dense pull transformations, correspondence support
and confidence maps. Reports work locally without uploading images. Manifest and
review files can contain identifying metadata and source paths; keep the complete
prepared directory private along with the images.

## Annotate in Slicer

On macOS, launch the helper from this checkout:

```sh
./Start\ DSA\ Workflow.command outputs/my-run
```

For other systems, load the helper in Slicer's Python Interactor as described in
the [Slicer guide](slicer/README.md). One ordinary segmentation stays fixed while
you browse frames. For local deformation, mark anatomical review and explicitly
record reviewed reference regions before contributing annotations.

Snapshots include the mask, editable segmentation and provenance. Local-registration
snapshots also include reviewed-coverage and unknown-region masks. Unknown pixels
must not be treated as negative training labels. Complete vessel-tree coverage
is not inferred automatically.

## Additional commands

```sh
dsa-preprocess inspect /path/to/organized-dataset
dsa-preprocess evaluate outputs/my-run --template -o outputs/my-run-landmarks.json
dsa-preprocess evaluate outputs/my-run --landmarks /path/to/reviewed-points.json \
  -o outputs/my-run-evaluation.json
dsa-preprocess batch /path/to/organized-dataset --config-dir configs \
  --limit 3 -o outputs/my-batch
```

Dataset discovery selects `*_angio.nrrd` and reads available sidecars. Configurations
for `batch` are named `<run_id>.json`, as shown by `inspect`. Known scanner summary
frames are excluded by default; undocumented summary frames require manual review.
Each successful batch run gets its own MinIP and report. Failed runs are recorded
in `batch_summary.json`, and the command returns a nonzero status.

The `pilot` command executes a predeclared comparison packet with a bounded fit
budget. See the [local alignment guide](docs/deformable-alignment.md). Independent
landmarks must be kept separate from fitting constraints. Numerical targets remain
provisional and require anatomical review and reader-repeatability assessment.

## Development and data handling

See [development and publication checks](docs/development.md). Tests use synthetic
images generated at runtime. Source data, annotations, generated reports, local
configuration, environments and research logs are ignored by Git and retained on
disk. Only generic examples belong in `examples/`.

This software prepares 2D projection annotations. It does not reconstruct 3D motion,
recover structures absent from the images, or correct blur already present within
a source frame. In-plane spacing is preserved, but placeholder or detector spacing
must not be interpreted as calibrated vessel-plane measurements.

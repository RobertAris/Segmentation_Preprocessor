# Local vessel alignment and anatomical review

An explicit `registration` configuration enables spatially varying alignment and
manifest schema 2. Existing configurations retain their translation behavior and
schema 1. New results are **candidates for anatomical review**, not automatically
accepted alignments. Contrast changes, missing branches and overlapping projected
vessels can make correspondence unresolved.

## Prepare a candidate

Run from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m dsa_preprocessor prepare SOURCE_angio.nrrd \
  --config MY_CONFIG.json --registration-method bspline -o outputs/NEW_RUN
```

This illustrative example is for a 320×256 acquisition; choose its frames and vessel regions
after reviewing the actual sequence:

```json
{
  "reference_frame": 0,
  "frames": [0, 1, 2],
  "polarity": "dark",
  "registration": {
    "method": "bspline",
    "global_model": "translation",
    "regions": [[20, 20, 280, 200]],
    "exclude_regions": [],
    "pyramid_scales": [2, 1],
    "grid_spacings_px": [128, 64],
    "regularization": 1.0,
    "max_displacement_px": 32.0
  }
}
```

Regions are `[x,y,width,height]` in original full-resolution pixels. Cover the
contributing branches across the tree. Exclusions prevent matching and accepting
support at those locations in both reference and source images. Registration filtering never replaces
the original intensity images used for annotation.

Methods are `translation`, `rigid`, `similarity`, `affine`, `bspline`, `tvl1` and
`demons`. The first four fit global models to local image correspondences.
B-splines fit a regularized local field. TV-L1 and Demons are feature-driven
comparators with external support gating, not fully masked force optimizers.
Demons requires the optional SimpleITK dependency; missing dependencies produce
an explicit error rather than a silent substitution. Full normalized settings are
saved in the preparation. Unknown settings and conflicting legacy manual shifts
are rejected.

Install the optional Demons backend with `uv pip install --python .venv/bin/python
'SimpleITK>=2.4,<3'`, or install the package's `registration` extra. Coarse-to-fine
matching is explicit: `pyramid_scales: [2,1]` uses half and full resolution;
the default `[1]` uses full resolution. Coarse matches guide the search, but every
accepted proposal still requires reciprocal evidence at the finest scale.

`registration.fitting_landmarks` is keyed by original source frame, with paired
`reference` and `moving` arrays of `[x,y]` points. These are fitting constraints,
not independent evaluation points. The reference geometry cannot be silently
changed under an existing annotation. Always choose a new output directory outside
the source dataset.

## Explicit keyframe paths

For a difficult long-range comparison, specify intermediate original frames:

```json
{
  "reference_frame": 90,
  "frames": [90, 105, 130, 160],
  "registration": {
    "method": "bspline",
    "paths": {"130": [105], "160": [105, 130]},
    "check_paths_directly": true,
    "max_path_disagreement_px": 2.0
  }
}
```

This estimates 90→105→130 and 90→105→130→160 in pull-map direction, then
checks 130 and 160 against independent direct-reference registrations. It uses
five unique directed fits. Every intermediate must be selected; cycles and
mixing reference-based fitting landmarks with nonempty paths are rejected.
Edges are cached with their field and support hashes. Composition propagates
all interpolation contributors, limits confidence to the least confident edge,
and rechecks the final geometry and inverse. A failed edge or unsupported direct
check cannot establish a valid path. The disagreement limit is a numerical
screening setting, not an anatomical acceptance tolerance.

Paths are explicit and do not assume known frame timing or a smooth physiological
trajectory. They do not jointly optimize the sequence. Changing the reference in
the report clears its old paths and fitting landmarks. Excluding an intermediate
drops paths that depend on it. For incompatible projections, prepare a separate
selected group in a new directory with its own reference and annotation.

## Saved transformations and support

| Artifact | Meaning |
| --- | --- |
| `registered.seq.nrrd` | Original intensities sampled once through the composed map |
| `reference.nrrd` | Original reference image and fixed geometry |
| `transforms/frame-000008-pull.npz` | Final reference-to-source displacement plus metadata/hash |
| `validity.seq.nrrd` | Valid original-image interpolation support |
| `correspondence.seq.nrrd` | Candidate shared-structure support requiring review |
| `confidence.seq.nrrd` | Uncalibrated matching score, not anatomical probability |
| `minip.nrrd` and validity | Exploratory projection restricted to candidate support |
| `minip.png` | Native PNG of the curated projection |
| `minip-comparison/index.html` | Automatic before/after viewer using the same selected frames |
| `minip-comparison/before.nrrd`, `after.nrrd` | Original/aligned all-frame projections with source-sampling validity |
| `report.html`, `review/` | Full-resolution source/result views and review controls |
| `manifest.json`, `config.json`, `transforms.tsv` | True source indices, configuration, source/code provenance and failures |

The float32 field `pull_displacement_yx` has shape `(2,height,width)`, components
`[dy,dx]` and mapping `Phi(y,x)=(y,x)+u(y,x)`. Coarse alignment is already included.
An old content shift `[dy,dx]` becomes field `[-dy,-dx]`. Scalar shifts in schema 2
describe coarse initialization only. Fields compose in coordinates; original
intensities are resampled once. The reference remains one spatial slice and the
sequence axis remains time/source-frame order.

Duplicates retain original indices and are not independent accuracy evidence.
Scanner summary images remain excluded. Folding, displacement, strain, inverse
support and correspondence checks can reject candidates. Manual review cannot
override a hard numerical failure.

Every successful preparation, including batch and pilot jobs, creates the
projection viewer automatically. The all-frame comparison retains the same
selected frames, including failures/exclusions, and omits only invalid sampling
after alignment. Its purpose is visual diagnosis; it does not grant anatomical
eligibility. The existing curated projection remains restricted to eligible
frames and correspondence support. Both use one linear display window without
added sharpening or denoising. Bright polarity uses a maximum projection, labeled
MIP; historical `minip` filenames remain compatible.

## Independent anatomical evaluation

Open the local `report.html`. Inspect peripheral branches at native resolution;
use blink, overlays, support and deformation views. The report collects explicit
frame decisions and corresponding points on the **original** reference/moving
images. Record branch identity, visibility, reference width and fitting/evaluation
role. Download the landmark JSON and run:

```sh
PYTHONPATH=src .venv/bin/python -m dsa_preprocessor evaluate outputs/NEW_RUN \
  --landmarks LANDMARKS.json -o outputs/NEW_RUN-landmark-evaluation.json
```

An empty collection packet is also available:

```sh
PYTHONPATH=src .venv/bin/python -m dsa_preprocessor evaluate outputs/NEW_RUN \
  --template -o outputs/NEW_RUN-landmark-template.json
```

Packets identify the exact preparation by fingerprint. Fitting-point reuse,
incorrect support-frame labels and transform provenance mismatches are rejected.
Older schema-1 preparations lack per-frame identity fingerprints. Their evaluator
verifies the original source file's SHA-256 and compares original frames to detect
reference copies; unavailable or mismatched originals prevent independent
evaluation. Schema-2 fingerprints support portable evaluation without originals.
Evaluation outputs record this identity-verification provenance and evaluator hash.
Primary errors are in the final reference plane through a checked inverse;
source-space residuals remain separate. Missing annotations/widths, unsupported
points and duplicate controls cannot establish passing accuracy. Report region
and worst errors as well as pooled statistics. Numerical targets remain
provisional and do not automatically approve anatomy or full-tree coverage.

## Slicer annotation and unknown regions

Load the prepared directory with the existing launcher. Schema 2 requires explicit
anatomical review before editing on a candidate frame. Record reviewed reference
regions in Slicer or import the report's downloaded review JSON. One ordinary
Segmentation node stays fixed while browsing time.

Export checks contributing frames and reviewed coverage. Positive labels outside
reviewed support must be resolved. Snapshots include the binary mask, editable
segmentation, reviewed-coverage mask, unknown mask and transform/review provenance.
Unknown pixels must not be used as negative training labels. Reference identity
alone does not establish complete vessel visibility. See the
[Slicer guide](../slicer/README.md) for controls and resume behavior.

## Bounded comparisons

```sh
PYTHONPATH=src .venv/bin/python -m dsa_preprocessor pilot PACKET.json \
  -o outputs/NEW_PILOT
```

A packet has `schema_version: 1`, `max_fits` at most 160 and `jobs`. Each job
specifies `job_id`, `source_file`, `purpose` and `config`. Purposes include
development, confirmation, reverse, ablation, retry, smoke and identity. Reverse
fits/retries consume the cap; numerical inversion is not an independent reverse
fit. Track the cumulative study budget across packets. The runner records actual
paths, hashes, environment, timing, cumulative process peak memory and failures;
it does not choose a winner from NCC or tune automatically.

Real-data acceptance requires independent correspondence
and human anatomical review; synthetic tests establish software behavior only.

# Development and publication

## Local verification

```sh
python -m pip install -e '.[test]'
python -m pytest -q
dsa-preprocess --help
```

Install `.[test,registration]` to exercise optional Demons. Tests generate synthetic
images in temporary directories and check registration, coordinate conventions,
validity, evaluation integrity, atomic outputs, projections and report behavior.
They do not establish anatomical accuracy.

GitHub CI tests Linux/Python 3.10 and 3.13 and macOS/Python 3.13, with a separate
optional-Demons job. It checks the installed CLI outside the checkout and builds
distributions. Hosted results are available after GitHub runs the workflow; local
checks do not establish the other platforms' results. CI uses the official
[checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python) actions with read-only
repository permissions and no publishing credentials.

## Build and inspect distributions

```sh
python -m pip install build
python -m build
python scripts/normalize_sdist.py dist/*.tar.gz
```

The wheel contains the Python package and CLI. Use the source checkout or source
distribution for the Slicer helper and guides. `MANIFEST.in` explicitly selects
public documentation and examples; local research notes and data must not enter
either distribution. Inspect both archive inventories before sharing.
The normalization step removes local owner/group identifiers from source-archive
headers while preserving file contents and executable permissions. Content checks
alone do not inspect this metadata.

Verify installation in a fresh environment outside the checkout, then run
`dsa-preprocess --help`. Importing only with `PYTHONPATH=src` does not establish
that the installed package works.

## Keep private material local

Keep acquisitions, derived images, annotations, review decisions, logs and reports
in ignored directories. Generated reports and manifests can contain identifying
images, source paths, hashes and metadata. `.gitignore` also excludes common
medical-image/data formats, local configuration, credentials, environments and
research handoffs. It does not remove files from disk.

Use `examples/` only for deliberately generic configurations. Use ignored
`configs/` for local runs. Generate synthetic fixtures in tests; do not add real
acquisitions or image-bearing reports as fixtures.

Before publishing, inspect candidate files and history:

```sh
git status --short
git diff --check
git ls-files
git log --oneline --all
```

Ignore rules do not untrack content or erase earlier commits. Untracking preserves
working files, but a clean publication history must also omit private historical
blobs. Keep existing research history locally and publish a separately reviewed
clean snapshot when necessary. Do not push all local branches or tags
indiscriminately. Ignoring some files does not prove that the remainder is suitable
for publication.

## Scientific and integration constraints

- Preserve original frame indices and time-axis semantics. Annotation uses one
  fixed 2D reference plane.
- Keep registration features separate from original intensities. Compose maps
  before resampling originals once; retain sampling and correspondence support.
- The diagnostic projection compares the same selected frames before/after;
  eligibility-filtered projections are labeled separately. Bright polarity uses
  maxima. Projection sharpness alone cannot establish correspondence accuracy.
- Separate fitting and independent evaluation points. Keep unknown or unsupported
  anatomy explicit in annotation exports.
- Prepare atomically in a new directory. Preserve sources and existing outputs
  when an operation fails.
- Review shared bifurcations and small branches before accepting real-data
  outputs. Synthetic tests establish no clinical performance or complete-tree claim.

Local coordination tools and research handoffs are optional and excluded from the
public package. Record experiment configuration, code/data versions and actual
output paths in private run records when performing research.
